import requests
from ddgs import DDGS

import folder as Folder
import metadata as Metadata
import verbose as Verbose

_TAG = "IMAGE SEEKER"


def Search(query, limit):

    image_number = Folder.Len()
    saved_images = 0
    seen_urls = set()

    max_attempts = limit * 20
    attempts = 0

    Verbose.Write(
        _TAG,
        f"Searching until {limit} images are saved...",
        Verbose.CONSOLE_STATUS.INFO,
    )

    with DDGS() as ddgs:
        while saved_images < limit and attempts < max_attempts:
            remaining = limit - saved_images
            batch_size = max(remaining * 5, 20)

            try:
                results = ddgs.images(query, max_results=batch_size)

                found_results = False

                for image in results:
                    found_results = True

                    if saved_images >= limit:
                        break

                    if attempts >= max_attempts:
                        break

                    attempts += 1

                    image_url = image.get("image")

                    if not image_url or image_url in seen_urls:
                        continue

                    seen_urls.add(image_url)

                    try:
                        image_response = requests.get(
                            image_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}
                        )

                        if image_response.status_code != 200:
                            continue

                        if not image_response.headers.get(
                            "Content-Type", ""
                        ).startswith("image/"):
                            continue

                        if Metadata.Find(image_response.url):
                            Verbose.Write(
                                _TAG,
                                "Skipping: Image already exists.",
                                Verbose.CONSOLE_STATUS.MESSAGE,
                            )
                            continue

                        file_name = f"image{image_number}.jpg"

                        if Folder.SaveFile(image_response.content, file_name):
                            Metadata.Write(image_response.url)

                            Verbose.Write(
                                "Image Seeker",
                                f"Saving image. [{file_name}]",
                                Verbose.CONSOLE_STATUS.MESSAGE,
                            )

                            image_number += 1
                            saved_images += 1

                    except requests.RequestException as error:
                        Verbose.Write(
                            _TAG,
                            f"Error: Download failed: {error}",
                            Verbose.CONSOLE_STATUS.ERROR,
                        )

                if not found_results:
                    break

            except Exception as error:
                Verbose.Write(
                    _TAG, f"Error: Search failed: {error}", Verbose.CONSOLE_STATUS.ERROR
                )
                break

    Verbose.Write(
        _TAG,
        f"Finished. Saved {saved_images} of {limit} images.",
        Verbose.CONSOLE_STATUS.SUCCESS,
    )
