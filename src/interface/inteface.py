import sys
from pathlib import Path

import src.helpers.verbose as Verbose

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

_TAG = "INTERFACE"

APP = QApplication(sys.argv)
UI_PATH = Path(__file__).parent / "design.ui"
UI_FILE = QFile(str(UI_PATH))

window = None

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

    return APP.exec()