import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    ProgressBar,
    RichLog,
    Static,
)
from textual.widgets.option_list import Option

import src.data.dataset as Dataset
import src.data.folder as Folder
import src.image.image_seeker as ImageSeeker
import src.data.metadata as Metadata
import src.model.model as Model
import src.interface.validate_window as ValidateWindow
import src.interface.inteface as Interface
import src.helpers.verbose as Verbose
import src.model.yolo as Yolo

_TAG = "MAIN"

APP_TITLE = "YOLO Pipeline"
SESSION_FILE = Path("data") / "SESSION.JSON"
DATASET_YAML = Path("dataset/data.yaml")
MODEL_GLOB = "runs/**/weights/best.pt"
DEFAULT_LIMIT = 100
DEFAULT_CONFIDENCE = 0.25
IMAGE_EXTENSIONS = set(Interface.IMAGE_EXTENSIONS)

MUTED = "#8A90A2"
ACCENT = "#5B8CFF"
GOOD = "#1F9D6B"
WARN = "#E0A030"
BAD = "#E5534B"

MENU = (
    (
        "COLLECT",
        (
            ("1", "Search Images", "Download images by keyword"),
            ("2", "Annotate", "Draw and review labels"),
            ("3", "Target Folder", "Choose the working folder"),
        ),
    ),
    (
        "TRAIN",
        (
            ("4", "Build Dataset", "Convert labels to YOLO format"),
            ("5", "Train", "Train the YOLO model"),
        ),
    ),
    (
        "USE MODEL",
        (
            ("6", "Webcam", "Live detection"),
            ("7", "Detect Image", "Run the model on a picture"),
        ),
    ),
    (
        "MAINTENANCE",
        (
            ("8", "Enhance Data", "Improve the collected images"),
            ("9", "Erase All", "Delete all data and models"),
            ("0", "Exit", "Close the program"),
        ),
    ),
)

ACTIONS = {
    "1": "DoSearch",
    "2": "DoAnnotate",
    "3": "DoSetFolder",
    "4": "DoBuildDataset",
    "5": "DoTrain",
    "6": "DoWebcam",
    "7": "DoDetect",
    "8": "DoEnhance",
    "9": "DoErase",
}

STEP_HINTS = {
    "3": "choose a target folder",
    "1": "search for images",
    "2": "annotate your images",
    "4": "build the dataset",
    "5": "train the model",
}

INPUT_TYPES = {"text": "text", "int": "integer", "float": "number"}


class ActionFailed(Exception):
    pass


def SafeCall(function, default=None):
    try:
        return function()
    except Exception:
        return default


def HumanSize(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"

        size /= 1024


def HumanTime(seconds):
    seconds = int(seconds)

    if seconds < 60:
        return f"{seconds}s"

    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"

    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def ListImages(folder):
    if not folder:
        return []

    return [
        file
        for file in Folder.GetFolder(folder).iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def ReadMetadata():
    if hasattr(Metadata, "Read"):
        return Metadata.Read()

    return list(getattr(Metadata, "metadata_content_buffer", None) or [])


def AnnotationStats(folder):
    key = Interface.NormalizeFolder(folder) if folder else None
    images = set()
    classes = set()
    boxes = 0
    urls = 0

    for line in ReadMetadata():
        if line.startswith(("http://", "https://")):
            urls += 1
            continue

        parts = line.split("|")

        if key is None or len(parts) < 3:
            continue

        if Interface.NormalizeFolder(parts[0]) != key:
            continue

        boxes += 1
        images.add(parts[4] if len(parts) > 4 and parts[4] else parts[1])

        if len(parts) > 3 and parts[3].isdigit():
            classes.add(int(parts[3]))

    return len(images), boxes, len(classes), urls


def DatasetStats():
    if not DATASET_YAML.exists():
        return None

    root = DATASET_YAML.parent
    images = 0
    labels = 0

    for file in root.rglob("*"):
        if file.suffix.lower() in IMAGE_EXTENSIONS:
            images += 1
        elif file.suffix.lower() == ".txt":
            labels += 1

    return {"images": images, "labels": labels, "modified": DATASET_YAML.stat().st_mtime}


def FindModel():
    configured = SafeCall(lambda: Path(Model.MODEL_PATH))

    if configured is not None and configured.exists():
        return configured

    found = [path for path in Path(".").glob(MODEL_GLOB) if path.is_file()]

    if not found:
        return None

    return max(found, key=lambda path: path.stat().st_mtime)


def SyncModelPath(path):
    configured = SafeCall(lambda: Path(Model.MODEL_PATH))

    if path is not None and configured != path:
        Model.MODEL_PATH = path
        Model._model = None


def GpuName():
    torch = sys.modules.get("torch")

    if torch is None:
        return "unknown"

    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)} (CUDA)"

    return "CPU only"


def CollectStatus(folder):
    images = SafeCall(lambda: ListImages(folder), [])
    annotated, boxes, classes, urls = SafeCall(
        lambda: AnnotationStats(folder), (0, 0, 0, 0)
    )
    model = SafeCall(FindModel)

    return {
        "folder": folder,
        "images": len(images),
        "image_bytes": sum(SafeCall(lambda f=f: f.stat().st_size, 0) for f in images),
        "annotated": min(annotated, len(images)) if images else annotated,
        "boxes": boxes,
        "classes": classes,
        "urls": urls,
        "dataset": SafeCall(DatasetStats),
        "model": model,
        "model_stat": SafeCall(lambda: model.stat()) if model else None,
        "gpu": SafeCall(GpuName, "unknown"),
    }


def BuildSteps(status):
    images = status["images"]
    annotated = status["annotated"]

    if images and annotated >= images:
        label_state = "done"
    elif annotated:
        label_state = "partial"
    else:
        label_state = "todo"

    return [
        (
            "done" if status["folder"] else "todo",
            "Target folder",
            status["folder"] or "not chosen yet",
            "3",
        ),
        (
            "done" if images else "todo",
            "Images collected",
            f"{images} images" if images else "none yet",
            "1",
        ),
        (
            label_state,
            "Images labeled",
            f"{annotated} of {images} images  ·  {status['boxes']} boxes",
            "2",
        ),
        (
            "done" if status["dataset"] else "todo",
            "Dataset built",
            "ready" if status["dataset"] else "not built yet",
            "4",
        ),
        (
            "done" if status["model"] else "todo",
            "Model trained",
            "ready" if status["model"] else "not trained yet",
            "5",
        ),
    ]


def ProjectRenderable(status):
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=MUTED)
    grid.add_column()

    folder = status["folder"]
    grid.add_row(
        "Folder", Text(folder or "not chosen", style="bold" if folder else WARN)
    )
    grid.add_row(
        "Images",
        f"{status['images']} files  ·  {HumanSize(status['image_bytes'])}",
    )
    grid.add_row("Labeled", f"{status['annotated']} of {status['images']} images")
    grid.add_row("Boxes", str(status["boxes"]))
    grid.add_row("Classes used", str(status["classes"]))
    grid.add_row("Registered URLs", str(status["urls"]))

    return grid


def ModelRenderable(status):
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=MUTED)
    grid.add_column()

    model = status["model"]

    if model is None:
        grid.add_row("Model", Text("not trained yet", style=WARN))
    else:
        grid.add_row("Model", Text(f"{model.parent.parent.name} / {model.name}", style="bold"))

        stat = status["model_stat"]

        if stat is not None:
            grid.add_row(
                "Trained",
                f"{datetime.fromtimestamp(stat.st_mtime):%Y-%m-%d %H:%M}  ·  {HumanSize(stat.st_size)}",
            )

    dataset = status["dataset"]

    if dataset is None:
        grid.add_row("Dataset", Text("not built yet", style=WARN))
    else:
        grid.add_row(
            "Dataset",
            f"{dataset['images']} images  ·  {dataset['labels']} label files",
        )

    grid.add_row("Device", status["gpu"])

    epochs = getattr(Yolo, "EPOCHS", None)
    size = getattr(Yolo, "IMAGE_SIZE", None)
    batch = getattr(Yolo, "BATCH_SIZE", None)

    if epochs and size and batch:
        grid.add_row("Training", f"{epochs} epochs  ·  {size}px  ·  batch {batch}")

    return grid


def PipelineRenderable(status):
    steps = BuildSteps(status)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=1)
    grid.add_column(min_width=18)
    grid.add_column()

    icons = {
        "done": Text("●", style=GOOD),
        "partial": Text("◐", style=WARN),
        "todo": Text("○", style=MUTED),
    }

    for state, name, detail, _ in steps:
        grid.add_row(
            icons[state],
            Text(name, style="bold" if state != "todo" else MUTED),
            Text(detail, style="" if state != "todo" else MUTED),
        )

    pending = next((step for step in steps if step[0] != "done"), None)

    grid.add_row("", "", "")

    if pending is None:
        grid.add_row(
            Text("→", style=ACCENT),
            Text("All set", style="bold"),
            Text("Press 6 or 7 to try your model, or 2 to keep labeling.", style=MUTED),
        )
    else:
        grid.add_row(
            Text("→", style=ACCENT),
            Text("Next step", style=f"bold {ACCENT}"),
            Text(f"Press {pending[3]}  ·  {STEP_HINTS[pending[3]]}"),
        )

    return grid


def OpenFile(path):
    if sys.platform.startswith("win"):
        os.startfile(str(path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class FormScreen(ModalScreen):

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, title, fields, submit="OK", subtitle=""):
        super().__init__()
        self.dialog_title = title
        self.fields = fields
        self.submit_label = submit
        self.subtitle = subtitle

    def compose(self):
        with Vertical(id="dialog"):
            yield Label(self.dialog_title, id="dialog_title")

            if self.subtitle:
                yield Label(self.subtitle, id="dialog_subtitle")

            for field in self.fields:
                yield Label(field["label"], classes="field_label")
                yield Input(
                    value=str(field.get("value", "")),
                    placeholder=field.get("placeholder", ""),
                    type=INPUT_TYPES[field.get("kind", "text")],
                    id=f"field_{field['key']}",
                )

            yield Label("", id="dialog_error")

            with Horizontal(id="dialog_buttons"):
                yield Button("Cancel", id="cancel")
                yield Button(self.submit_label, variant="primary", id="ok")

    def on_mount(self):
        self.query(Input).first().focus()

    def action_cancel(self):
        self.dismiss(None)

    @on(Button.Pressed, "#cancel")
    def CancelPressed(self):
        self.dismiss(None)

    @on(Button.Pressed, "#ok")
    def OkPressed(self):
        self.Submit()

    @on(Input.Submitted)
    def InputSubmitted(self, event):
        inputs = list(self.query(Input))

        if event.input is inputs[-1]:
            self.Submit()
        else:
            inputs[inputs.index(event.input) + 1].focus()

    def Submit(self):
        values = {}

        for field in self.fields:
            widget = self.query_one(f"#field_{field['key']}", Input)
            raw = widget.value.strip()

            if not raw:
                if field.get("required"):
                    self.Fail(f"{field['label']} is required.", widget)
                    return

                values[field["key"]] = None
                continue

            kind = field.get("kind", "text")

            try:
                value = int(raw) if kind == "int" else float(raw) if kind == "float" else raw
            except ValueError:
                self.Fail(f"{field['label']} must be a number.", widget)
                return

            if kind != "text":
                low = field.get("min")
                high = field.get("max")

                if (low is not None and value < low) or (high is not None and value > high):
                    self.Fail(
                        f"{field['label']} must be between {low} and {high}.", widget
                    )
                    return

            values[field["key"]] = value

        self.dismiss(values)

    def Fail(self, message, widget):
        self.query_one("#dialog_error", Label).update(message)
        widget.focus()


class ConfirmScreen(ModalScreen):

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, title, message, confirm="Confirm", phrase=None, danger=False):
        super().__init__()
        self.dialog_title = title
        self.message = message
        self.confirm_label = confirm
        self.phrase = phrase
        self.danger = danger

    def compose(self):
        with Vertical(id="dialog"):
            yield Label(self.dialog_title, id="dialog_title")
            yield Static(self.message, id="dialog_message")

            if self.phrase:
                yield Label(f"Type {self.phrase} to confirm", classes="field_label")
                yield Input(placeholder=self.phrase, id="phrase_input")

            yield Label("", id="dialog_error")

            with Horizontal(id="dialog_buttons"):
                yield Button("Cancel", id="cancel")
                yield Button(
                    self.confirm_label,
                    variant="error" if self.danger else "primary",
                    id="ok",
                )

    def on_mount(self):
        if self.phrase:
            self.query_one("#phrase_input", Input).focus()
        else:
            self.query_one("#ok", Button).focus()

    def action_cancel(self):
        self.dismiss(False)

    @on(Button.Pressed, "#cancel")
    def CancelPressed(self):
        self.dismiss(False)

    @on(Input.Submitted)
    def InputSubmitted(self):
        self.Confirm()

    @on(Button.Pressed, "#ok")
    def OkPressed(self):
        self.Confirm()

    def Confirm(self):
        if self.phrase:
            typed = self.query_one("#phrase_input", Input).value.strip()

            if typed != self.phrase:
                self.query_one("#dialog_error", Label).update(
                    f"Type exactly {self.phrase} to continue."
                )
                return

        self.dismiss(True)


class ImageTree(DirectoryTree):

    def filter_paths(self, paths):
        return [path for path in paths if self.Visible(path)]

    def Visible(self, path):
        if path.name.startswith("."):
            return False

        try:
            return path.is_dir() or path.suffix.lower() in IMAGE_EXTENSIONS
        except OSError:
            return False


class FilePickerScreen(ModalScreen):

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, title="Choose an image"):
        super().__init__()
        self.dialog_title = title

    def compose(self):
        with Vertical(id="dialog"):
            yield Label(self.dialog_title, id="dialog_title")
            yield Input(value=str(Path.cwd()), id="path_input")
            yield ImageTree(str(Path.cwd()), id="file_tree")
            yield Label("", id="dialog_error")

            with Horizontal(id="dialog_buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Open", variant="primary", id="ok")

    def action_cancel(self):
        self.dismiss(None)

    @on(Button.Pressed, "#cancel")
    def CancelPressed(self):
        self.dismiss(None)

    @on(DirectoryTree.FileSelected)
    def FileSelected(self, event):
        self.query_one("#path_input", Input).value = str(event.path)

    @on(Input.Submitted, "#path_input")
    def PathSubmitted(self, event):
        path = Path(event.value.strip().strip('"'))

        if path.is_dir():
            self.query_one("#file_tree", ImageTree).path = path
        else:
            self.Choose()

    @on(Button.Pressed, "#ok")
    def OkPressed(self):
        self.Choose()

    def Choose(self):
        path = Path(self.query_one("#path_input", Input).value.strip().strip('"'))

        if not path.is_file():
            self.query_one("#dialog_error", Label).update(
                "Select an image file from the list or type its path."
            )
            return

        self.dismiss(str(path))


class PipelineApp(App):

    TITLE = APP_TITLE

    CSS = """
    Screen {
        background: #0F1117;
        color: #E4E6EB;
    }

    Header {
        background: #181A20;
        color: #E4E6EB;
    }

    #body {
        height: 1fr;
    }

    #sidebar {
        width: 40;
        background: #181A20;
        border-right: solid #2A2E39;
    }

    #menu {
        background: #181A20;
        border: none;
        padding: 1 1;
        height: 100%;
        scrollbar-size-vertical: 1;
        scrollbar-background: #181A20;
        scrollbar-color: #444957;
    }

    #menu > .option-list--option {
        padding: 0 1;
    }

    #menu > .option-list--option-highlighted {
        background: #1F3A7A;
        color: #FFFFFF;
        text-style: none;
    }

    #menu > .option-list--option-hover {
        background: #252832;
    }

    #menu > .option-list--option-disabled {
        color: #6C7285;
    }

    #main {
        padding: 1 2;
    }

    #top_row {
        height: auto;
    }

    .panel {
        border: round #2A2E39;
        border-title-color: #8A90A2;
        border-title-style: bold;
        background: #14161C;
        padding: 0 1;
        margin-bottom: 1;
        height: auto;
        scrollbar-background: #14161C;
        scrollbar-color: #444957;
    }

    #project_box {
        width: 1fr;
        margin-right: 1;
    }

    #model_box {
        width: 1fr;
    }

    #progress {
        margin-top: 1;
        width: 100%;
    }

    #activity {
        height: 1fr;
        min-height: 8;
    }

    FormScreen, ConfirmScreen, FilePickerScreen {
        align: center middle;
        background: #00000099;
    }

    #dialog {
        width: 72;
        height: auto;
        max-height: 90%;
        background: #181A20;
        border: round #2F6BFF;
        padding: 1 2;
    }

    #dialog_title {
        text-style: bold;
        color: #FFFFFF;
    }

    #dialog_subtitle {
        color: #8A90A2;
    }

    #dialog_message {
        margin-top: 1;
        height: auto;
    }

    .field_label {
        color: #8A90A2;
        margin-top: 1;
    }

    #dialog_error {
        color: #E5534B;
        height: auto;
        margin-top: 1;
    }

    #dialog_buttons {
        height: auto;
        margin-top: 1;
        align-horizontal: right;
    }

    #dialog_buttons Button {
        margin-left: 1;
    }

    #file_tree {
        height: 16;
        margin-top: 1;
        border: round #2A2E39;
    }
    """

    BINDINGS = [Binding(str(number), f"choose('{number}')", show=False) for number in range(10)] + [
        Binding("r", "refresh_status", "Refresh"),
        Binding("c", "clear_log", "Clear log"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.folder_name = None
        self.busy = False

    def compose(self):
        yield Header(show_clock=True, icon="◆")

        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield OptionList(*self.MenuOptions(), id="menu")

            with VerticalScroll(id="main"):
                with Horizontal(id="top_row"):
                    with Vertical(id="project_box", classes="panel"):
                        yield Static(id="project")
                        yield ProgressBar(
                            total=100, show_eta=False, id="progress"
                        )

                    with Vertical(id="model_box", classes="panel"):
                        yield Static(id="model")

                yield Static(id="pipeline", classes="panel")
                yield RichLog(id="activity", classes="panel", wrap=True)

        yield Footer()

    def MenuOptions(self):
        options = []

        for index, (group, items) in enumerate(MENU):
            if index:
                options.append(None)

            options.append(
                Option(Text(group, style=f"bold {MUTED}"), disabled=True)
            )

            for number, label, description in items:
                options.append(
                    Option(
                        Text.assemble(
                            (f" {number} ", f"bold {ACCENT}"),
                            (f" {label}", "bold"),
                            "\n",
                            (f"     {description}", MUTED),
                        ),
                        id=number,
                    )
                )

        return options

    def on_mount(self):
        self.query_one("#project_box").border_title = "Project"
        self.query_one("#model_box").border_title = "Model"
        self.query_one("#pipeline").border_title = "Pipeline"
        self.query_one("#activity").border_title = "Activity"
        self.query_one("#menu").focus()

        self.RestoreSession()
        self.Log("Ready. Use the arrows and Enter, press a number, or open Ctrl+P for every command.")
        self.RefreshStatus()

    def get_system_commands(self, screen):
        yield from super().get_system_commands(screen)

        for _, items in MENU:
            for number, label, description in items:
                yield SystemCommand(
                    f"{number} · {label}",
                    description,
                    partial(self.action_choose, number),
                )

    def Log(self, message, kind="info"):
        icons = {
            "info": ("•", MUTED),
            "ok": ("✔", GOOD),
            "error": ("✖", BAD),
            "warn": ("!", WARN),
        }

        icon, color = icons[kind]

        self.query_one("#activity", RichLog).write(
            Text.assemble(
                (f"{datetime.now():%H:%M:%S}", MUTED),
                "  ",
                (icon, color),
                "  ",
                message,
            )
        )

    def RestoreSession(self):
        try:
            data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
            self.folder_name = data.get("folder") or None
        except (OSError, ValueError):
            self.folder_name = None

    def SaveSession(self):
        try:
            SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
            SESSION_FILE.write_text(
                json.dumps({"folder": self.folder_name}), encoding="utf-8"
            )
        except OSError:
            pass

    @work(thread=True, exclusive=True, group="status")
    def RefreshStatus(self):
        status = CollectStatus(self.folder_name)
        self.call_from_thread(self.ApplyStatus, status)

    def ApplyStatus(self, status):
        SyncModelPath(status["model"])

        self.sub_title = f"target: {status['folder']}" if status["folder"] else "no target folder"
        self.query_one("#project", Static).update(ProjectRenderable(status))
        self.query_one("#model", Static).update(ModelRenderable(status))
        self.query_one("#pipeline", Static).update(PipelineRenderable(status))
        self.query_one("#progress", ProgressBar).update(
            total=max(status["images"], 1), progress=status["annotated"]
        )

    def action_refresh_status(self):
        self.RefreshStatus()
        self.notify("Dashboard refreshed.", timeout=2)

    def action_clear_log(self):
        self.query_one("#activity", RichLog).clear()

    def action_quit_app(self):
        if not self.busy and len(self.screen_stack) == 1:
            self.exit()

    def action_choose(self, number):
        if self.busy or len(self.screen_stack) > 1:
            return

        if number == "0":
            self.exit()
            return

        if number in ACTIONS:
            self.RunAction(number)

    @on(OptionList.OptionSelected, "#menu")
    def MenuSelected(self, event):
        self.action_choose(event.option.id)

    @work(exclusive=True, group="action")
    async def RunAction(self, number):
        self.busy = True

        try:
            await getattr(self, ACTIONS[number])()
        finally:
            self.busy = False
            self.RefreshStatus()

    def RunInTerminal(self, title, function, pause=True, prepare=True):
        if prepare:
            SafeCall(lambda: Folder.Configure(""))

        started = time.monotonic()
        result = None
        error = None

        can_suspend = getattr(self._driver, "can_suspend", False)
        context = self.suspend() if can_suspend else contextlib.nullcontext()

        with context:
            console = Console()
            console.rule(f"[bold]{title}[/bold]")

            try:
                result = function()
            except KeyboardInterrupt:
                error = "Interrupted by user"
            except ActionFailed as failure:
                error = str(failure)
            except Exception as exception:
                traceback.print_exc()
                error = f"{type(exception).__name__}: {exception}"

            console.rule(style="dim")

            if error:
                console.print(f"[red]✖ {error}[/red]")

            if pause or error:
                console.print("Press Enter to return to the dashboard...", style="dim")

                try:
                    input()
                except EOFError:
                    pass

        elapsed = HumanTime(time.monotonic() - started)

        if error:
            self.Log(f"{title} failed: {error}", "error")
        else:
            self.Log(f"{title} finished in {elapsed}", "ok")

        return result, error

    async def RequireFolder(self):
        if not self.folder_name:
            self.notify("Choose a target folder first.", severity="warning")
            await self.DoSetFolder()

        return bool(self.folder_name)

    def ApplyFolder(self, name):
        self.folder_name = name.strip()
        Folder.Configure(self.folder_name)
        self.SaveSession()

    async def DoSetFolder(self):
        values = await self.push_screen_wait(
            FormScreen(
                "Target Folder",
                [
                    {
                        "key": "folder",
                        "label": "Folder name",
                        "value": self.folder_name or "",
                        "placeholder": "e.g. cat",
                        "required": True,
                    }
                ],
                submit="Select",
                subtitle="All images and labels are grouped under this name.",
            )
        )

        if not values:
            return

        try:
            self.ApplyFolder(values["folder"])
        except Exception as error:
            self.Log(f"Could not select the folder: {error}", "error")
            return

        self.Log(f"Target folder set to {self.folder_name}", "ok")

    async def DoSearch(self):
        values = await self.push_screen_wait(
            FormScreen(
                "Search Images",
                [
                    {
                        "key": "folder",
                        "label": "Target folder",
                        "value": self.folder_name or "",
                        "placeholder": "e.g. cat",
                        "required": True,
                    },
                    {
                        "key": "keyword",
                        "label": "Search keywords",
                        "placeholder": "what the images should show",
                        "required": True,
                    },
                    {
                        "key": "limit",
                        "label": "How many images to save",
                        "value": DEFAULT_LIMIT,
                        "kind": "int",
                        "min": 1,
                        "max": 1000000,
                        "required": True,
                    },
                ],
                submit="Start",
                subtitle="Downloads images matching the keywords into the target folder.",
            )
        )

        if not values:
            return

        try:
            self.ApplyFolder(values["folder"])
        except Exception as error:
            self.Log(f"Could not select the folder: {error}", "error")
            return

        keyword = values["keyword"]
        limit = values["limit"]

        result, error = self.RunInTerminal(
            f"Search Images  ·  {keyword}",
            lambda: ImageSeeker.Search(keyword, limit),
            prepare=False,
        )

        if not error and isinstance(result, int):
            self.Log(f"{result} of {limit} images saved to {self.folder_name}", "ok")

    async def DoAnnotate(self):
        if not await self.RequireFolder():
            return

        def Run():
            Console().print("Annotator open. Close its window to return to the dashboard.", style="dim")

            if Interface.Start(self.folder_name):
                raise ActionFailed("The annotator could not start. See the messages above.")

        self.RunInTerminal("Annotate", Run, pause=False)

    async def DoBuildDataset(self):
        if not await self.RequireFolder():
            return

        self.RunInTerminal(
            "Build Dataset", lambda: Dataset.GenerateDataset(self.folder_name)
        )

    async def DoTrain(self):
        dataset = Path(getattr(Yolo, "DATASET", DATASET_YAML))

        if not dataset.exists():
            self.notify("Build the dataset first (option 4).", severity="error")
            return

        epochs = getattr(Yolo, "EPOCHS", "?")
        size = getattr(Yolo, "IMAGE_SIZE", "?")
        batch = getattr(Yolo, "BATCH_SIZE", "?")
        base = getattr(Yolo, "MODEL", "?")

        confirmed = await self.push_screen_wait(
            ConfirmScreen(
                "Train model",
                f"Base model:  [b]{base}[/b]\n"
                f"Epochs:      [b]{epochs}[/b]\n"
                f"Image size:  [b]{size}px[/b]\n"
                f"Batch size:  [b]{batch}[/b]\n"
                f"Device:      [b]{GpuName()}[/b]\n\n"
                "Training can take a long time. Press Ctrl+C in the terminal to stop it.",
                confirm="Start training",
            )
        )

        if confirmed:
            self.RunInTerminal("Train", Yolo.Train)

    def RequireModel(self):
        if FindModel() is None:
            self.notify("No trained model found. Train one first (option 5).", severity="error")
            return False

        SyncModelPath(FindModel())

        return True

    async def DoWebcam(self):
        if not self.RequireModel():
            return

        def Run():
            Console().print("Close the video window or press Q to stop.", style="dim")
            Model.LoadModel()
            Model.DetectWebcam()

        self.RunInTerminal("Webcam", Run, pause=False)

    async def DoDetect(self):
        if not self.RequireModel():
            return

        path = await self.push_screen_wait(FilePickerScreen("Choose the image to analyze"))

        if not path:
            return

        values = await self.push_screen_wait(
            FormScreen(
                "Detection settings",
                [
                    {
                        "key": "conf",
                        "label": "Minimum confidence (0 - 1)",
                        "value": DEFAULT_CONFIDENCE,
                        "kind": "float",
                        "min": 0.01,
                        "max": 1,
                        "required": True,
                    }
                ],
                submit="Detect",
                subtitle=path,
            )
        )

        if not values:
            return

        started = time.time()
        output = Path(getattr(Model, "OUTPUT_FOLDER", "runs/detect/predict")) / f"{Path(path).stem}_detected.jpg"

        def Run():
            Model.LoadModel()

            try:
                Model.DetectImage(path, conf=values["conf"])
            finally:
                if output.exists() and output.stat().st_mtime >= started:
                    Console().print(f"Result saved at {output}", style="dim")
                    SafeCall(lambda: OpenFile(output))

        self.RunInTerminal(f"Detect  ·  {Path(path).name}", Run, pause=False)

    async def DoEnhance(self):
        if not await self.RequireFolder():
            return

        self.RunInTerminal(
            "Enhance Data", lambda: ValidateWindow.EnhanceData(self.folder_name)
        )

    async def DoErase(self):
        confirmed = await self.push_screen_wait(
            ConfirmScreen(
                "Erase everything",
                "This permanently deletes:\n\n"
                "  • all downloaded images\n"
                "  • every label and the metadata registry\n"
                "  • the built dataset\n"
                "  • trained models and training runs\n\n"
                "[b]This cannot be undone.[/b]",
                confirm="Erase all",
                phrase="ERASE",
                danger=True,
            )
        )

        if not confirmed:
            return

        def Run():
            Metadata.EraseAll()
            Metadata.Close()
            Folder.EraseAll()
            Folder.Delete("runs")
            Folder.Delete("data")
            Folder.Delete("dataset")

            os.makedirs("data", exist_ok=True)
            Metadata.metadata_file_buffer = None
            Folder.Configure("")
            Metadata.Configure()

        _, error = self.RunInTerminal("Erase All Data", Run)

        if not error:
            self.folder_name = None
            self.Log("All data erased.", "warn")


def Main():
    Folder.Configure("")
    Metadata.Configure()

    try:
        PipelineApp().run()
    finally:
        SafeCall(Metadata.Close)
        SafeCall(Verbose.Clear)


if __name__ == "__main__":
    Main()