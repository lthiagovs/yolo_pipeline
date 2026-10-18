import ast

import cv2 as ValidateWindow

import folder as Folder
import metadata as Metadata
import verbose as Verbose

_TAG = "VALIDATION_WINDOW"

# GLOBALS
drawning = False
start_point = None
end_point = None
boxes = []


def DrawBoxes(image):
    image = image.copy()

    for start, end in boxes:
        ValidateWindow.rectangle(image, start, end, (0, 255, 0), 2)

    ValidateWindow.imshow("Images", image)


def MouseEvent(event, x, y, flags, param):

    global drawning, start_point, end_point
    image = param.copy()

    # DRAW ALL BOXES
    for start, end in boxes:
        ValidateWindow.rectangle(image, start, end, (0, 255, 0), 2)

    # REGISTER FIRST CLICK TO DRAW BOX
    if event == ValidateWindow.EVENT_LBUTTONDOWN:
        drawning = True
        start_point = (x, y)

    # REGISTER DRAG
    if drawning:
        ValidateWindow.rectangle(image, start_point, (x, y), (0, 255, 0), 2)
        ValidateWindow.imshow("Images", image)

    # REGISTER AND CREATE THE BOX
    if event == ValidateWindow.EVENT_LBUTTONUP and drawning:
        drawning = False
        end_point = (x, y)
        boxes.append((start_point, end_point))
        DrawBoxes(param)


def Open(folder):

    # SET WINDOW
    ValidateWindow.namedWindow("Images", ValidateWindow.WINDOW_NORMAL)
    ValidateWindow.resizeWindow("Images", 800, 600)
    ValidateWindow.moveWindow("Images", 460, 190)

    # TODO START IN THE LAST NUMBER IN METADATA
    file_number = 0

    # SELECT IMAGES
    for file in Folder.GetFolder(folder).iterdir():
        # CLEAR BOX REGISTER
        boxes.clear()

        if file.is_file():
            image = ValidateWindow.imread(str(file))

            if image is None:
                continue

            ValidateWindow.setMouseCallback("Images", MouseEvent, image)
            ValidateWindow.imshow("Images", image)

            key = 0

            while key != 13:
                key = ValidateWindow.waitKey(0)
                if key == 27 and boxes:
                    boxes.pop()
                    DrawBoxes(image)

            for box in boxes:
                Metadata.Write(str(folder) + "|" + str(file_number) + "|" + str(box))

            file_number = file_number + 1

    ValidateWindow.destroyAllWindows()


# Enhance Data
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

        image = ValidateWindow.imread(str(file))

        if image is None:
            continue

        height, width = image.shape[:2]

        transformations = [
            (
                "grayscale",
                ValidateWindow.cvtColor(image, ValidateWindow.COLOR_BGR2GRAY),
            ),
            ("flip_h", ValidateWindow.flip(image, 1)),
            ("flip_v", ValidateWindow.flip(image, 0)),
            ("rotate_180", ValidateWindow.rotate(image, ValidateWindow.ROTATE_180)),
            ("brightness", ValidateWindow.convertScaleAbs(image, alpha=1.2, beta=20)),
        ]

        for operation, augmented in transformations:
            new_boxes = [TransformBox(box, width, height, operation) for box in boxes]

            if len(augmented.shape) == 2:
                augmented = ValidateWindow.cvtColor(
                    augmented, ValidateWindow.COLOR_GRAY2BGR
                )

            success, encoded = ValidateWindow.imencode(".jpg", augmented)

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
