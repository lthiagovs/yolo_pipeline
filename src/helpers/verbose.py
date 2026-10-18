import os
from enum import Enum

from colorama import Fore, just_fix_windows_console

_TAG = "VERBOSE"

just_fix_windows_console()

verbose_on = True


class CONSOLE_STATUS(Enum):
    ERROR = Fore.RED
    SUCCESS = Fore.GREEN
    INFO = Fore.CYAN
    WARNING = Fore.YELLOW
    MESSAGE = Fore.WHITE


def Write(module_name: str, message: str, type: CONSOLE_STATUS):
    if verbose_on:
        print(type.value + module_name + "\t" + " >> " + "\t" + message)


def Clear():
    os.system("cls" if os.name == "nt" else "clear")
