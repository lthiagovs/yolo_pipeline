import json
import os
import queue
import random
import re
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from html import unescape
from html.parser import HTMLParser
from urllib import robotparser
from urllib.parse import urldefrag, urljoin, urlparse

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

import src.data.folder as Folder
import src.data.metadata as Metadata
import src.helpers.verbose as Verbose

_TAG = "IMAGE SEEKER"

WORKERS = 24
MIN_SIDE = 200
MAX_SIDE = 2048
MAX_ASPECT = 6.0
MIN_BYTES = 2048
MAX_BYTES = 30 * 1024 * 1024
MIN_CONTRAST = 8.0
JPEG_QUALITY = 95
DUPLICATE_DISTANCE = 4

CONNECT_TIMEOUT = 6
READ_TIMEOUT = 20
REQUEST_RETRIES = 3

QUEUE_SIZE = 4000
MAX_ATTEMPTS_FACTOR = 60
MAX_PAGES_PER_QUERY = 40
EMPTY_QUERY_LIMIT = 3
PROGRESS_EVERY = 10

DDG_RESULTS = 500
WIKIMEDIA_WIDTH = 1600

CRAWL_DEPTH = 1
CRAWL_PAGES_PER_SITE = 12
CRAWL_LINKS_PER_PAGE = 200
CRAWL_SEEDS_PER_QUERY = 15
CRAWL_PAGE_BYTES = 3 * 1024 * 1024
MIN_HINT_SIZE = 120

HASH_FILE_NAME = "IMAGE_HASHES.DATA"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SKIP_EXTENSIONS = (".svg", ".gif", ".ico")
PAGE_SKIP_EXTENSIONS = (".pdf", ".zip", ".css", ".js", ".mp4", ".mp3", ".xml", ".json")
SKIP_WORDS = (
    "sprite",
    "icon",
    "logo",
    "avatar",
    "favicon",
    "pixel",
    "badge",
    "button",
    "banner",
    "placeholder",
)

BING_URL = "https://www.bing.com/images/async"
GOOGLE_URL = "https://www.google.com/search"
WIKIMEDIA_URL = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_URL = "https://api.openverse.org/v1/images/"
PIXABAY_URL = "https://pixabay.com/api/"
PEXELS_URL = "https://api.pexels.com/v1/search"
FLICKR_URL = "https://www.flickr.com/services/rest/"

WIKIMEDIA_AGENT = "YoloPipelineImageSeeker/1.0 (dataset building; python-requests)"

SOURCE_DELAY = {
    "bing": 0.8,
    "google": 2.0,
    "duckduckgo": 1.5,
    "wikimedia": 0.4,
    "openverse": 0.6,
    "pixabay": 0.5,
    "pexels": 0.5,
    "flickr": 0.5,
    "crawler": 0.4,
}

SOURCE_KEYS = {
    "pixabay": "PIXABAY_API_KEY",
    "pexels": "PEXELS_API_KEY",
    "flickr": "FLICKR_API_KEY",
}

USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
)

MODIFIERS = (
    "",
    "photo",
    "photograph",
    "high resolution",
    "hd",
    "close up",
    "outdoor",
    "indoor",
    "in nature",
    "in the wild",
    "city",
    "street",
    "white background",
    "dark",
    "daylight",
    "night",
    "from above",
    "side view",
    "front view",
    "back view",
    "group",
    "single",
    "small",
    "large",
    "red",
    "blue",
    "green",
    "black",
    "white",
    "brown",
    "yellow",
    "vintage",
    "modern",
    "real life",
    "stock photo",
    "wallpaper",
)

GOOGLE_IMAGE_PATTERN = re.compile(r'\["(https?:[^"]+?)",\s*(\d+),\s*(\d+)\]')
BING_IMAGE_PATTERN = re.compile(r"murl&quot;:&quot;(.*?)&quot;")
GOOGLE_HOSTS = ("google.com", "gstatic.com", "googleusercontent.com", "ggpht.com")
POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)

THREAD_LOCAL = threading.local()
ROBOTS = {}
ROBOTS_LOCK = threading.Lock()


def NewSession(pool):
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool)

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def GetSession():
    session = getattr(THREAD_LOCAL, "session", None)

    if session is None:
        session = NewSession(8)
        THREAD_LOCAL.session = session

    return session


def RandomHeaders(referer=None, accept="*/*"):
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": accept,
        "Accept-Language": "en-US,en;q=0.9",
    }

    if referer:
        headers["Referer"] = referer

    return headers


def Origin(url):
    parts = urlparse(url)

    return f"{parts.scheme}://{parts.netloc}/"


def NormalizeUrl(url):
    return urldefrag(url.strip())[0]


def RetryDelay(response, attempt):
    value = response.headers.get("Retry-After", "")

    if value.isdigit():
        return min(int(value), 30)

    return min(2 ** (attempt + 1), 20)


class HttpClient:

    def __init__(self, delay):
        self.delay = delay
        self.session = NewSession(4)
        self.last_request = 0.0

    def Wait(self):
        remaining = self.delay - (time.monotonic() - self.last_request)

        if remaining > 0:
            time.sleep(remaining)

        self.last_request = time.monotonic()

    def Get(self, url, params=None, headers=None, accept="*/*"):
        merged = RandomHeaders(accept=accept)

        if headers:
            merged.update(headers)

        for attempt in range(REQUEST_RETRIES):
            self.Wait()

            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=merged,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                )
            except requests.RequestException:
                time.sleep(2**attempt)
                continue

            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(RetryDelay(response, attempt))
                continue

            return response

        return None


class HashStore:

    def __init__(self, path):
        self.path = path
        self.array = np.zeros(1024, dtype=np.uint64)
        self.count = 0

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        self.Load()

    def Load(self):
        if not os.path.exists(self.path):
            return

        with open(self.path, encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                try:
                    self.Append(int(line, 16))
                except ValueError:
                    continue

    def Append(self, value):
        if self.count == len(self.array):
            self.array = np.concatenate(
                [self.array, np.zeros(len(self.array), dtype=np.uint64)]
            )

        self.array[self.count] = value
        self.count += 1

    def IsDuplicate(self, value):
        if self.count == 0:
            return False

        difference = self.array[: self.count] ^ np.uint64(value)
        distance = POPCOUNT[difference.view(np.uint8)].reshape(-1, 8).sum(axis=1)

        return bool((distance <= DUPLICATE_DISTANCE).any())

    def Add(self, value):
        self.Append(value)

        with open(self.path, "a", encoding="utf-8") as file:
            file.write(f"{value:016x}\n")


def DecodeEscapes(text):
    try:
        return json.loads(f'"{text}"')
    except ValueError:
        return text


def SearchBing(query, client):
    for page in range(MAX_PAGES_PER_QUERY):
        response = client.Get(
            BING_URL,
            params={"q": query, "first": page * 35, "count": 35, "mmasync": 1},
            accept="text/html",
        )

        if response is None or response.status_code != 200:
            return

        urls = BING_IMAGE_PATTERN.findall(response.text)

        if not urls:
            return

        for url in dict.fromkeys(urls):
            yield unescape(url), None


def SearchGoogle(query, client):
    for page in range(MAX_PAGES_PER_QUERY):
        response = client.Get(
            GOOGLE_URL,
            params={
                "q": query,
                "tbm": "isch",
                "hl": "en",
                "start": page * 20,
                "ijn": page,
            },
            accept="text/html",
        )

        if response is None or response.status_code != 200:
            return

        found = False

        for raw, width, height in GOOGLE_IMAGE_PATTERN.findall(response.text):
            url = DecodeEscapes(raw)
            host = urlparse(url).netloc.lower()

            if host.endswith(GOOGLE_HOSTS):
                continue

            if min(int(width), int(height)) < MIN_SIDE:
                continue

            found = True
            yield url, None

        if not found:
            return


def SearchDuckDuckGo(query, client):
    if DDGS is None:
        return

    client.Wait()

    with DDGS() as ddgs:
        for result in ddgs.images(query, max_results=DDG_RESULTS):
            url = result.get("image")

            if not url:
                continue

            width = result.get("width") or 0
            height = result.get("height") or 0

            if width and height and min(width, height) < MIN_SIDE:
                continue

            yield url, result.get("url")


def SearchWikimedia(query, client):
    offset = None

    for _ in range(MAX_PAGES_PER_QUERY):
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": query,
            "gsrlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "iiurlwidth": WIKIMEDIA_WIDTH,
        }

        if offset is not None:
            params["gsroffset"] = offset

        response = client.Get(
            WIKIMEDIA_URL, params=params, headers={"User-Agent": WIKIMEDIA_AGENT}
        )

        if response is None or response.status_code != 200:
            return

        data = response.json()
        pages = data.get("query", {}).get("pages", {}).values()

        for page in pages:
            info = (page.get("imageinfo") or [{}])[0]

            if info.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                continue

            url = info.get("thumburl") or info.get("url")

            if url:
                yield url, info.get("descriptionurl")

        offset = data.get("continue", {}).get("gsroffset")

        if offset is None:
            return


def SearchOpenverse(query, client):
    for page in range(1, MAX_PAGES_PER_QUERY + 1):
        response = client.Get(
            OPENVERSE_URL, params={"q": query, "page": page, "page_size": 20}
        )

        if response is None or response.status_code != 200:
            return

        results = response.json().get("results", [])

        if not results:
            return

        for result in results:
            url = result.get("url")

            if url:
                yield url, result.get("foreign_landing_url")


def SearchPixabay(query, client):
    key = os.environ.get(SOURCE_KEYS["pixabay"])

    for page in range(1, MAX_PAGES_PER_QUERY + 1):
        response = client.Get(
            PIXABAY_URL,
            params={
                "key": key,
                "q": query,
                "image_type": "photo",
                "per_page": 200,
                "page": page,
            },
        )

        if response is None or response.status_code != 200:
            return

        hits = response.json().get("hits", [])

        if not hits:
            return

        for hit in hits:
            url = hit.get("largeImageURL") or hit.get("webformatURL")

            if url:
                yield url, hit.get("pageURL")


def SearchPexels(query, client):
    key = os.environ.get(SOURCE_KEYS["pexels"])

    for page in range(1, MAX_PAGES_PER_QUERY + 1):
        response = client.Get(
            PEXELS_URL,
            params={"query": query, "per_page": 80, "page": page},
            headers={"Authorization": key},
        )

        if response is None or response.status_code != 200:
            return

        photos = response.json().get("photos", [])

        if not photos:
            return

        for photo in photos:
            source = photo.get("src", {})
            url = source.get("large2x") or source.get("large") or source.get("original")

            if url:
                yield url, photo.get("url")


def SearchFlickr(query, client):
    key = os.environ.get(SOURCE_KEYS["flickr"])

    for page in range(1, MAX_PAGES_PER_QUERY + 1):
        response = client.Get(
            FLICKR_URL,
            params={
                "method": "flickr.photos.search",
                "api_key": key,
                "text": query,
                "media": "photos",
                "content_type": 1,
                "sort": "relevance",
                "extras": "url_l,url_c,url_o",
                "per_page": 500,
                "page": page,
                "format": "json",
                "nojsoncallback": 1,
            },
        )

        if response is None or response.status_code != 200:
            return

        photos = response.json().get("photos", {})

        for photo in photos.get("photo", []):
            url = photo.get("url_l") or photo.get("url_c") or photo.get("url_o")

            if url:
                yield url, None

        if page >= int(photos.get("pages", 0)):
            return


def ToInt(value):
    digits = re.sub(r"\D", "", value or "")

    return int(digits) if digits else 0


def SmallHint(values):
    width = ToInt(values.get("width"))
    height = ToInt(values.get("height"))

    return bool(
        (width and width < MIN_HINT_SIZE) or (height and height < MIN_HINT_SIZE)
    )


def BestFromSrcset(text):
    if not text:
        return None

    best = None
    best_score = -1.0

    for part in text.split(","):
        pieces = part.strip().split()

        if not pieces:
            continue

        score = 0.0

        if len(pieces) > 1:
            try:
                score = float(pieces[1][:-1])
            except ValueError:
                score = 0.0

        if score > best_score:
            best = pieces[0]
            best_score = score

    return best


class PageParser(HTMLParser):

    def __init__(self, base):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.images = []
        self.links = []

    def AddImage(self, value):
        if not value:
            return

        value = value.strip()

        if not value or value.startswith("data:"):
            return

        url = urljoin(self.base, value)

        if urlparse(url).scheme in ("http", "https"):
            self.images.append(url)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)

        if tag == "img":
            if SmallHint(values):
                return

            source = BestFromSrcset(values.get("srcset") or values.get("data-srcset"))

            if not source:
                for name in ("data-src", "data-original", "data-lazy-src", "src"):
                    if values.get(name):
                        source = values[name]
                        break

            self.AddImage(source)

        elif tag == "source":
            self.AddImage(BestFromSrcset(values.get("srcset") or values.get("data-srcset")))

        elif tag == "meta":
            if values.get("property") in ("og:image", "twitter:image") or values.get(
                "name"
            ) == "twitter:image":
                self.AddImage(values.get("content"))

        elif tag == "link":
            if values.get("rel") == "image_src":
                self.AddImage(values.get("href"))

        elif tag == "a":
            href = values.get("href")

            if not href:
                return

            url = urljoin(self.base, href.strip())
            path = urlparse(url).path.lower()

            if path.endswith(IMAGE_EXTENSIONS):
                self.AddImage(url)
            elif not path.endswith(PAGE_SKIP_EXTENSIONS):
                self.links.append(url)


def LooksLikeImage(url):
    lowered = url.lower()
    path = urlparse(lowered).path

    if path.endswith(SKIP_EXTENSIONS):
        return False

    return not any(word in lowered for word in SKIP_WORDS)


def RobotsAllowed(url, client):
    parts = urlparse(url)
    root = f"{parts.scheme}://{parts.netloc}"

    with ROBOTS_LOCK:
        known = root in ROBOTS
        parser = ROBOTS.get(root)

    if not known:
        response = client.Get(root + "/robots.txt")
        parser = None

        if response is not None and response.status_code == 200:
            parser = robotparser.RobotFileParser()
            parser.parse(response.text.splitlines())

        with ROBOTS_LOCK:
            ROBOTS[root] = parser

    return parser is None or parser.can_fetch("*", url)


def FetchPage(client, url):
    response = client.Get(url, accept="text/html,application/xhtml+xml")

    if response is None or response.status_code != 200:
        return None

    if "html" not in response.headers.get("Content-Type", "").lower():
        return None

    return response.content[:CRAWL_PAGE_BYTES].decode(
        response.encoding or "utf-8", errors="replace"
    )


def CrawlSite(seed, client, visited):
    host = urlparse(seed).netloc
    pending = deque([(seed, 0)])
    emitted = set()
    pages = 0

    while pending and pages < CRAWL_PAGES_PER_SITE:
        page, depth = pending.popleft()
        key = NormalizeUrl(page)

        if key in visited:
            continue

        visited.add(key)

        if not RobotsAllowed(page, client):
            continue

        html = FetchPage(client, page)

        if html is None:
            continue

        pages += 1
        parser = PageParser(page)
        parser.feed(html)
        parser.close()

        for image in parser.images:
            if image in emitted or not LooksLikeImage(image):
                continue

            emitted.add(image)
            yield image, page

        if depth >= CRAWL_DEPTH:
            continue

        for link in parser.links[:CRAWL_LINKS_PER_PAGE]:
            if urlparse(link).netloc == host and NormalizeUrl(link) not in visited:
                pending.append((link, depth + 1))


def FindPages(query):
    if DDGS is None:
        return []

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=CRAWL_SEEDS_PER_QUERY)

        return [result["href"] for result in results if result.get("href")]
    except Exception:
        return []


def MakeCrawler(seeds):
    state = {"seeds_pending": list(seeds or [])}
    visited = set()

    def Crawl(query, client):
        pages = state["seeds_pending"]
        state["seeds_pending"] = []
        pages = pages + FindPages(query)

        for page in pages:
            yield from CrawlSite(page, client, visited)

    return Crawl


def BuildSources(names, seeds):
    registry = {
        "bing": SearchBing,
        "duckduckgo": SearchDuckDuckGo,
        "wikimedia": SearchWikimedia,
        "openverse": SearchOpenverse,
        "pixabay": SearchPixabay,
        "pexels": SearchPexels,
        "flickr": SearchFlickr,
        "crawler": MakeCrawler(seeds),
        "google": SearchGoogle,
    }

    selected = {}

    for name in names or registry:
        function = registry.get(name)

        if function is None:
            Verbose.Write(
                _TAG, f"Unknown source ignored: {name}", Verbose.CONSOLE_STATUS.ERROR
            )
            continue

        key = SOURCE_KEYS.get(name)

        if key and not os.environ.get(key):
            if names:
                Verbose.Write(
                    _TAG,
                    f"{name} skipped: set the {key} environment variable.",
                    Verbose.CONSOLE_STATUS.ERROR,
                )
            continue

        if name == "duckduckgo" and DDGS is None:
            Verbose.Write(
                _TAG,
                "duckduckgo skipped: package ddgs is not installed.",
                Verbose.CONSOLE_STATUS.ERROR,
            )
            continue

        selected[name] = function

    return selected


def ExpandQueries(query, modifiers):
    bases = [query] if isinstance(query, str) else list(query)
    expanded = []

    for base in bases:
        for modifier in modifiers:
            expanded.append(f"{base} {modifier}".strip())

    return list(dict.fromkeys(expanded))


def Put(candidates, item, stop):
    while not stop.is_set():
        try:
            candidates.put(item, timeout=0.5)
            return True
        except queue.Full:
            continue

    return False


def Produce(name, function, queries, candidates, stop, stats):
    client = HttpClient(SOURCE_DELAY.get(name, 0.5))
    empty_streak = 0

    for query in queries:
        if stop.is_set():
            return

        produced = 0

        try:
            for url, referer in function(query, client):
                if not Put(candidates, (name, url, referer), stop):
                    return

                produced += 1
                stats["produced"][name] += 1
        except Exception as error:
            Verbose.Write(
                _TAG,
                f"Error: {name} failed on '{query}': {error}",
                Verbose.CONSOLE_STATUS.ERROR,
            )

        empty_streak = empty_streak + 1 if produced == 0 else 0

        if empty_streak >= EMPTY_QUERY_LIMIT:
            Verbose.Write(
                _TAG,
                f"{name} returned nothing {EMPTY_QUERY_LIMIT} times in a row. Disabled.",
                Verbose.CONSOLE_STATUS.ERROR,
            )
            return


def ProcessImage(data):
    if len(data) < MIN_BYTES:
        return None

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return None

    height, width = image.shape[:2]
    shortest = min(height, width)
    longest = max(height, width)

    if shortest < MIN_SIDE or longest / shortest > MAX_ASPECT:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if gray.std() < MIN_CONTRAST:
        return None

    tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (tiny[:, 1:] > tiny[:, :-1]).flatten()
    digest = int.from_bytes(np.packbits(bits).tobytes(), "big")

    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    success, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )

    if not success:
        return None

    return encoded.tobytes(), digest


def FetchImage(name, url, referer):
    headers = RandomHeaders(
        referer or Origin(url), "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
    )

    with GetSession().get(
        url,
        headers=headers,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        stream=True,
    ) as response:
        if response.status_code != 200:
            return None

        kind = response.headers.get("Content-Type", "").split(";")[0].strip().lower()

        if "svg" in kind or not (
            kind.startswith("image/") or kind == "application/octet-stream"
        ):
            return None

        length = response.headers.get("Content-Length", "")

        if length.isdigit() and int(length) > MAX_BYTES:
            return None

        chunks = []
        total = 0

        for chunk in response.iter_content(65536):
            total += len(chunk)

            if total > MAX_BYTES:
                return None

            chunks.append(chunk)

        final_url = response.url

    processed = ProcessImage(b"".join(chunks))

    if processed is None:
        return None

    content, digest = processed

    return {
        "candidate": url,
        "url": final_url,
        "content": content,
        "hash": digest,
    }


def Download(name, url, referer):
    try:
        return FetchImage(name, url, referer)
    except Exception:
        return None


def LoadKnownUrls():
    return {
        NormalizeUrl(line)
        for line in Metadata.Read()
        if line.startswith(("http://", "https://"))
    }


def Search(query, limit, sources=None, workers=WORKERS, seeds=None, modifiers=None):

    selected = BuildSources(sources, seeds)

    if not selected:
        Verbose.Write(_TAG, "Error: No source available.", Verbose.CONSOLE_STATUS.ERROR)
        return 0

    queries = ExpandQueries(query, modifiers or MODIFIERS)
    hashes = HashStore(
        os.path.join(os.path.dirname(Metadata.METADATA_FILE), HASH_FILE_NAME)
    )
    known = LoadKnownUrls()
    image_number = Folder.Len()
    saved = 0
    attempts = 0
    max_attempts = limit * MAX_ATTEMPTS_FACTOR

    stats = {
        "produced": Counter(),
        "saved": Counter(),
        "rejected": Counter(),
        "duplicates": Counter(),
    }

    Verbose.Write(
        _TAG,
        f"Searching until {limit} images are saved "
        f"({len(selected)} sources, {len(queries)} queries, {workers} workers)...",
        Verbose.CONSOLE_STATUS.INFO,
    )

    stop = threading.Event()
    candidates = queue.Queue(QUEUE_SIZE)

    producers = [
        threading.Thread(
            target=Produce,
            args=(name, function, queries, candidates, stop, stats),
            daemon=True,
        )
        for name, function in selected.items()
    ]

    for producer in producers:
        producer.start()

    pool = ThreadPoolExecutor(max_workers=workers)
    pending = {}
    started = time.monotonic()

    try:
        while saved < limit and attempts < max_attempts:
            while len(pending) < workers * 2 and attempts < max_attempts:
                try:
                    name, url, referer = candidates.get(
                        timeout=0.05 if pending else 0.5
                    )
                except queue.Empty:
                    break

                key = NormalizeUrl(url)

                if key in known:
                    stats["duplicates"][name] += 1
                    continue

                known.add(key)
                attempts += 1
                pending[pool.submit(Download, name, url, referer)] = (name, key)

            if not pending:
                if not any(p.is_alive() for p in producers) and candidates.empty():
                    break

                continue

            done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)

            for future in done:
                name, key = pending.pop(future)
                result = future.result()

                if result is None:
                    stats["rejected"][name] += 1
                    continue

                final_key = NormalizeUrl(result["url"])

                if final_key != key and final_key in known:
                    stats["duplicates"][name] += 1
                    continue

                if hashes.IsDuplicate(result["hash"]):
                    stats["duplicates"][name] += 1
                    continue

                file_name = f"image{image_number}.jpg"

                if not Folder.SaveFile(result["content"], file_name):
                    stats["rejected"][name] += 1
                    continue

                known.add(final_key)
                hashes.Add(result["hash"])
                Metadata.Write(result["url"])

                image_number += 1
                saved += 1
                stats["saved"][name] += 1

                if saved % PROGRESS_EVERY == 0 or saved == limit:
                    rate = saved / max(time.monotonic() - started, 0.001)
                    Verbose.Write(
                        _TAG,
                        f"Saved {saved}/{limit} ({rate:.1f} images/s) [{file_name}]",
                        Verbose.CONSOLE_STATUS.MESSAGE,
                    )

                if saved >= limit:
                    break
    except KeyboardInterrupt:
        Verbose.Write(_TAG, "Interrupted by user.", Verbose.CONSOLE_STATUS.ERROR)
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)

    for name in selected:
        Verbose.Write(
            _TAG,
            f"{name}: {stats['produced'][name]} found, {stats['saved'][name]} saved, "
            f"{stats['rejected'][name]} rejected, {stats['duplicates'][name]} duplicates",
            Verbose.CONSOLE_STATUS.INFO,
        )

    Verbose.Write(
        _TAG,
        f"Finished. Saved {saved} of {limit} images.",
        Verbose.CONSOLE_STATUS.SUCCESS,
    )

    return saved