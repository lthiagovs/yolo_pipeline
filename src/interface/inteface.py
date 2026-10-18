import sys
from pathlib import Path

import src.helpers.verbose as Verbose

from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QComboBox, QLabel
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap
from PySide6.QtCore import QFile, Qt

_TAG = "INTERFACE"

APP = QApplication(sys.argv)
UI_PATH = Path(__file__).parent / "design.ui"
UI_FILE = QFile(str(UI_PATH))

window = None
image_files = []

def Start():

    global window

    if not UI_FILE.open(QFile.OpenModeFlag.ReadOnly):
        Verbose.Write(_TAG, "Error to initialize interface", Verbose.CONSOLE_STATUS.ERROR)
        return 1

    window = QUiLoader().load(UI_FILE)
    UI_FILE.close()

    if window is None:
        Verbose.Write(_TAG, "Error to load UI file", Verbose.CONSOLE_STATUS.ERROR)
        return 1

    window.show()

    Configure()
    return APP.exec()

def Configure():
    button = window.findChild(QPushButton, "next_button")
    button.clicked.connect(Next)

# UI Helpers:

def SetImage(image_path: str):
    global window

    frame = window.findChild(QWidget, "image_frame")

    if frame is None:
        return

    if not hasattr(frame, "image_label"):
        frame.image_label = QLabel(frame)

        frame.image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        frame.image_label.setGeometry(frame.rect())

    pixmap = QPixmap(image_path)

    if pixmap.isNull():
        return

    frame.image_label.setPixmap(
        pixmap.scaled(
            frame.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
    )

    frame.image_label.show()

def Next():
    print("Botão Next pressionado!")

def Back():
    print("Botão Next pressionado!")

def PopulateClasses(classes: list[str]):
    dropdown = window.findChild(QComboBox, "class_dropdown")

    if dropdown is None:
        return

    dropdown.clear()
    dropdown.addItems(classes)

def GetSelectedClass():
    dropdown = window.findChild(QComboBox, "class_dropdown")
    return dropdown.currentText()

def SetLabelText(label_name: str, text: str):
    label = window.findChild(QLabel, label_name)

    if label is not None:
        label.setText(text)