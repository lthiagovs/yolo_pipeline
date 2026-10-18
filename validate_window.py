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
