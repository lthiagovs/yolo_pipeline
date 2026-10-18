import shutil
from pathlib import Path

import src.helpers.verbose as Verbose

_TAG = "FOLDER"

BASE_FOLDER = "data\\"
folder = None


# CONFIG
def Configure(folder_name):

    global folder

    folder = Path(BASE_FOLDER + folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    Verbose.Write(_TAG, "Creating folder.", Verbose.CONSOLE_STATUS.SUCCESS)

    return True


def Delete(folder_name):
    if Path(folder_name).exists():
        shutil.rmtree(folder_name)


def Len():
    if folder == None or not folder:
        Verbose.Write(
            _TAG, "Error: Folder does not exist...", Verbose.CONSOLE_STATUS.ERROR
        )
        return 0

    return sum(1 for file in folder.iterdir() if file.is_file())


def Get():
    return folder


def GetFolder(folder):
    return Path(BASE_FOLDER + folder)


def SaveFile(content, file_name):
    if folder == None or not folder:
        Verbose.Write(
            _TAG, "Error: Folder does not exist...", Verbose.CONSOLE_STATUS.ERROR
        )
        return False

    if not content:
        Verbose.Write(_TAG, "Error: Null file content...", Verbose.CONSOLE_STATUS.ERROR)
        return False

    new_file = Path(folder) / str(file_name)
    new_file.write_bytes(content)
    Verbose.Write(_TAG, "File saved.", Verbose.CONSOLE_STATUS.SUCCESS)

    return True


def ListAllFoldersIn(folder: str) -> list[Path]:
    return [p for p in Path(folder).iterdir() if p.is_dir()]


def EraseAll():

    global folder

    shutil.rmtree("data")
    folder = None
    return True
