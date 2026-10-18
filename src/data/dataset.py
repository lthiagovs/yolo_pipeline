import ast
import json
import random
import shutil
from pathlib import Path

import cv2

import src.data.folder as Folder
import src.data.metadata as Metadata
import src.helpers.verbose as Verbose

_TAG = "DATASET"

DATASET_FOLDER = Path("dataset")
SEED = 42

SPLITS = {"train": 0.70, "val": 0.20, "test": 0.10}

TILE = 1024
OVERLAP = 0.20
MAX_SIDE = 3072
MIN_VISIBLE = 0.40
KEEP_EMPTY = 0.25
CHECK_SAMPLES = 20


def LoadMetadata(folder_name):
    if Metadata.metadata_content_buffer is None or isinstance(
        Metadata.metadata_content_buffer, str
    ):
        raise RuntimeError("Run Metadata.Configure() first.")

    annotations = {}

    for line in Metadata.metadata_content_buffer:
        parts = line.split("|", 2)

        if len(parts) != 3:
            continue

        class_name, number, box = parts

        if class_name != folder_name:
            continue

        try:
            number = int(number)
        except ValueError:
            continue

        annotations.setdefault(number, []).append(box)

    return annotations


def ParseBox(box, scale):
    (x1, y1), (x2, y2) = ast.literal_eval(box)
    xa, xb = sorted((float(x1) * scale, float(x2) * scale))
    ya, yb = sorted((float(y1) * scale, float(y2) * scale))
    return xa, ya, xb, yb


def SplitFiles(files):
    if len(files) < 3:
        raise ValueError("Error: Need at least 3 images")

    files = files.copy()
    random.Random(SEED).shuffle(files)

    train_count = max(1, int(len(files) * SPLITS["train"]))
    val_count = max(1, int(len(files) * SPLITS["val"]))

    if train_count + val_count >= len(files):
        train_count = len(files) - 2
        val_count = 1

    return {
        "train": files[:train_count],
        "val": files[train_count : train_count + val_count],
        "test": files[train_count + val_count :],
    }


def TileStarts(size):
    if size <= TILE:
        return [0]

    stride = int(TILE * (1 - OVERLAP))
    starts = list(range(0, size - TILE, stride))
    starts.append(size - TILE)
    return starts


def WriteTiles(image, boxes, stem, split, rng):
    h0, w0 = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h0, w0))

    if scale < 1.0:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    h, w = image.shape[:2]

    pixel_boxes = []
    for box in boxes:
        try:
            pixel_boxes.append(ParseBox(box, scale))
        except (ValueError, SyntaxError, TypeError) as error:
            print(f"Invalid box on {stem}: {error}")

    img_dir = DATASET_FOLDER / "images" / split
    lbl_dir = DATASET_FOLDER / "labels" / split

    tiles_saved = 0
    boxes_saved = 0
    sizes = []

    for y0 in TileStarts(h):
        for x0 in TileStarts(w):
            tile = image[y0 : y0 + TILE, x0 : x0 + TILE]
            th, tw = tile.shape[:2]
            lines = []
            tile_sizes = []

            for xa, ya, xb, yb in pixel_boxes:
                ix1, iy1 = max(xa, x0), max(ya, y0)
                ix2, iy2 = min(xb, x0 + tw), min(yb, y0 + th)
                iw, ih = ix2 - ix1, iy2 - iy1

                if iw <= 1 or ih <= 1:
                    continue

                full_area = (xb - xa) * (yb - ya)
                if full_area <= 0:
                    continue
                if (iw * ih) / full_area < MIN_VISIBLE:
                    continue

                cx = ((ix1 + ix2) / 2 - x0) / tw
                cy = ((iy1 + iy2) / 2 - y0) / th
                lines.append(f"0 {cx:.6f} {cy:.6f} {iw / tw:.6f} {ih / th:.6f}\n")
                tile_sizes.append(max(iw, ih))

            if not lines and rng.random() > KEEP_EMPTY:
                continue

            name = f"{stem}_x{x0}_y{y0}"
            cv2.imwrite(
                str(img_dir / f"{name}.jpg"),
                tile,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            (lbl_dir / f"{name}.txt").write_text("".join(lines), encoding="utf-8")

            tiles_saved += 1
            boxes_saved += len(lines)
            sizes.extend(tile_sizes)

    return tiles_saved, boxes_saved, sizes


def GenerateYAML(folder_name):
    dataset_path = DATASET_FOLDER.resolve().as_posix()

    content = (
        f"path: {json.dumps(dataset_path)}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        f"  0: {json.dumps(folder_name)}\n"
    )

    (DATASET_FOLDER / "data.yaml").write_text(content, encoding="utf-8")


def CheckDataset(samples=CHECK_SAMPLES):
    check_dir = DATASET_FOLDER / "_check"
    check_dir.mkdir(parents=True, exist_ok=True)

    images = sorted((DATASET_FOLDER / "images" / "train").glob("*.jpg"))
    labeled = [
        img
        for img in images
        if (DATASET_FOLDER / "labels" / "train" / f"{img.stem}.txt")
        .read_text(encoding="utf-8")
        .strip()
    ]

    if not labeled:
        return

    chosen = random.Random(SEED).sample(labeled, min(samples, len(labeled)))

    for img_path in chosen:
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        label_path = DATASET_FOLDER / "labels" / "train" / f"{img_path.stem}.txt"

        for line in label_path.read_text(encoding="utf-8").splitlines():
            _, cx, cy, bw, bh = map(float, line.split())
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(str(check_dir / img_path.name), image)


def GenerateDataset(folder_name):
    annotations = LoadMetadata(folder_name)
    source_folder = Folder.GetFolder(folder_name)

    all_files = sorted(f for f in source_folder.iterdir() if f.is_file())

    indexed_files = []
    for file_count, file in enumerate(all_files):
        if cv2.imread(str(file)) is None:
            continue
        indexed_files.append((file_count, file))

    if len(indexed_files) < 3:
        raise ValueError("Need at least 3 images.")

    if DATASET_FOLDER.exists():
        shutil.rmtree(DATASET_FOLDER)

    for split in SPLITS:
        (DATASET_FOLDER / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_FOLDER / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_data = SplitFiles(indexed_files)
    rng = random.Random(SEED)

    all_sizes = []
    totals = {}

    for split, entries in split_data.items():
        tiles_total = 0
        boxes_total = 0

        for file_count, file in entries:
            image = cv2.imread(str(file))
            tiles, boxes, sizes = WriteTiles(
                image,
                annotations.get(file_count, []),
                f"img{file_count}",
                split,
                rng,
            )
            tiles_total += tiles
            boxes_total += boxes
            all_sizes.extend(sizes)

        totals[split] = (len(entries), tiles_total, boxes_total)

    GenerateYAML(folder_name)

    Verbose.Write(_TAG, " === Results === ", Verbose.CONSOLE_STATUS.INFO)
    for split, (n_img, n_tiles, n_boxes) in totals.items():
        Verbose.Write(
            _TAG,
            f"{split}: {n_img} images -> {n_tiles} tiles, {n_boxes} boxes",
            Verbose.CONSOLE_STATUS.INFO,
        )

    CheckDataset()


def EnhanceData(folder):

    source_folder = Folder.GetFolder(folder)

    files = sorted(
        (
            file
            for file in source_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
            and file.stem.startswith("image")
            and file.stem[5:].isdigit()
        ),
        key=lambda file: int(file.stem[5:]),
    )

    metadata = Metadata.metadata_content_buffer

    if metadata is None or isinstance(metadata, str):
        raise RuntimeError("Metadata is not configured.")

    annotations = {}

    for line in metadata:
        parts = line.split("|", 2)

        if len(parts) != 3 or parts[0] != folder:
            continue

        try:
            image_number = int(parts[1])
            box = ast.literal_eval(parts[2])
            annotations.setdefault(image_number, []).append(box)
        except (ValueError, SyntaxError):
            continue

    def TransformBox(box, width, height, operation):
        (x1, y1), (x2, y2) = box

        x_min, x_max = sorted((int(x1), int(x2)))
        y_min, y_max = sorted((int(y1), int(y2)))

        if operation in ("flip_h", "rotate_180"):
            x_min, x_max = width - 1 - x_max, width - 1 - x_min

        if operation in ("flip_v", "rotate_180"):
            y_min, y_max = height - 1 - y_max, height - 1 - y_min

        return (
            (max(0, x_min), max(0, y_min)),
            (min(width - 1, x_max), min(height - 1, y_max)),
        )

    next_number = max((int(file.stem[5:]) for file in files), default=-1) + 1

    created = 0

    for file in files:
        image_number = int(file.stem[5:])
        boxes = annotations.get(image_number, [])

        if not boxes:
            continue

        image = cv2.imread(str(file))

        if image is None:
            continue

        height, width = image.shape[:2]

        transformations = [
            (
                "grayscale",
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            ),
            ("flip_h", cv2.flip(image, 1)),
            ("flip_v", cv2.flip(image, 0)),
            ("rotate_180", cv2.rotate(image, cv2.ROTATE_180)),
            ("brightness", cv2.convertScaleAbs(image, alpha=1.2, beta=20)),
        ]

        for operation, augmented in transformations:
            new_boxes = [TransformBox(box, width, height, operation) for box in boxes]

            if len(augmented.shape) == 2:
                augmented = cv2.cvtColor(augmented, cv2.COLOR_GRAY2BGR)

            success, encoded = cv2.imencode(".jpg", augmented)

            if not success:
                continue

            new_name = f"image{next_number}.jpg"
            output_path = source_folder / new_name

            try:
                with open(output_path, "wb") as output_file:
                    output_file.write(encoded.tobytes())
            except OSError as error:
                Verbose.Write(
                    _TAG,
                    f"Failed to save {new_name}: {error}",
                    Verbose.CONSOLE_STATUS.ERROR,
                )
                continue

            for box in new_boxes:
                Metadata.Write(f"{folder}|{next_number}|{box}")

            Verbose.Write(
                _TAG, f"Saved {new_name} ({operation})", Verbose.CONSOLE_STATUS.SUCCESS
            )

            next_number += 1
            created += 1

    Verbose.Write(
        _TAG,
        f"Created {created} augmented images in {source_folder}.",
        Verbose.CONSOLE_STATUS.SUCCESS,
    )
