import src.data.dataset as Dataset
import src.data.folder as Folder
import src.image.image_seeker as ImageSeeker
import src.data.metadata as Metadata
import src.model.model as Model
import src.interface.validate_window as ValidateWindow
import src.helpers.verbose as Verbose
import src.model.yolo as Yolo

_TAG = "MAIN"

# GLOBALS
folder_name = None
url = None
image_limit = None
choice = 0


def WriteInfo():
    Verbose.Clear()
    if folder_name:
        Verbose.Write(
            _TAG, "TARGET FOLDER: " + folder_name, Verbose.CONSOLE_STATUS.INFO
        )
    Verbose.Write(_TAG, "0 - Exit", Verbose.CONSOLE_STATUS.MESSAGE)
    Verbose.Write(_TAG, "1 - Search Data", Verbose.CONSOLE_STATUS.MESSAGE)
    Verbose.Write(_TAG, "2 - Set Data", Verbose.CONSOLE_STATUS.MESSAGE)
    Verbose.Write(_TAG, "3 - Set Folder Name", Verbose.CONSOLE_STATUS.MESSAGE)
    Verbose.Write(_TAG, "4 - Build Dataset", Verbose.CONSOLE_STATUS.MESSAGE)
    Verbose.Write(_TAG, "5 - Train", Verbose.CONSOLE_STATUS.MESSAGE)
    Verbose.Write(_TAG, "6 - Use Model", Verbose.CONSOLE_STATUS.MESSAGE)
    Verbose.Write(_TAG, "7 - Use Model Target", Verbose.CONSOLE_STATUS.MESSAGE)
    Verbose.Write(_TAG, "8 - Enhance Data", Verbose.CONSOLE_STATUS.MESSAGE)
    Verbose.Write(_TAG, "9 - Erase All Data", Verbose.CONSOLE_STATUS.MESSAGE)
    Verbose.Write(_TAG, "Type choice: ", Verbose.CONSOLE_STATUS.INFO)


def BuildData():

    global folder_name, url, image_limit

    Verbose.Write(_TAG, "Starting...", Verbose.CONSOLE_STATUS.SUCCESS)
    Verbose.Write(_TAG, "Type folder name:", Verbose.CONSOLE_STATUS.INFO)

    folder_name = input()

    Folder.Configure(folder_name)

    Verbose.Write(_TAG, "Type images keywoard:", Verbose.CONSOLE_STATUS.INFO)
    url = input()

    Verbose.Write(_TAG, "Type search limit:", Verbose.CONSOLE_STATUS.INFO)
    image_limit = int(input())

    # REQUEST IMAGES ----------------------------------------------------------------

    ImageSeeker.Search(url, image_limit)


while True:
    Folder.Configure("")
    Metadata.Configure()

    # CHOICE ------------------------------------------------------------------------

    WriteInfo()

    choice = input()

    if not choice:
        continue

    choice = int(choice)

    if choice == 1:
        BuildData()

    elif choice == 2:
        if not folder_name:
            Verbose.Write(
                _TAG, "Error: Set Folder Name First!", Verbose.CONSOLE_STATUS.ERROR
            )
        ValidateWindow.Open(folder_name)

    elif choice == 3:
        folder_name = input()
        Folder.Configure(folder_name)

    elif choice == 4:
        if not folder_name:
            Verbose.Write(
                _TAG, "Error: Set Folder Name First!", Verbose.CONSOLE_STATUS.ERROR
            )
        Dataset.GenerateDataset(folder_name)

    elif choice == 5:
        Yolo.Train()

    elif choice == 6:
        Model.LoadModel()
        Model.DetectWebcam()

    elif choice == 7:
        Model.LoadModel()
        Verbose.Write(_TAG, "Type target path:", Verbose.CONSOLE_STATUS.INFO)
        target = input()
        Verbose.Write(_TAG, "Type conf value (float):", Verbose.CONSOLE_STATUS.INFO)
        conf_value = input()
        Model.DetectImage(target, conf=float(conf_value))

    elif choice == 8:
        if not folder_name:
            Verbose.Write(
                _TAG, "Error: Set Folder Name First!", Verbose.CONSOLE_STATUS.ERROR
            )
        ValidateWindow.EnhanceData(folder_name)

    elif choice == 9:
        Metadata.Close()
        Metadata.EraseAll()
        Folder.EraseAll()
        Folder.Delete("runs")
        Folder.Delete("data")
        Folder.Delete("dataset")

    elif choice == 0:
        break

    input()

# END ---------------------------------------------------------------------------
Verbose.Write(_TAG, "Press any key to continue...", Verbose.CONSOLE_STATUS.MESSAGE)
input()
Verbose.Clear()
