import ast
import os
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QLineEdit,
    QCheckBox,
    QVBoxLayout,
    QSizePolicy,
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import (
    QImage,
    QImageReader,
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QBrush,
    QFont,
    QFontMetrics,
    QLinearGradient,
)
from PySide6.QtCore import QFile, Qt, QRectF, QPointF, QSize

import src.helpers.verbose as Verbose
import src.data.folder as Folder
import src.data.metadata as Metadata

_TAG = "INTERFACE"

APP = QApplication(sys.argv)
UI_PATH = Path(__file__).parent / "design.ui"
UI_FILE = QFile(str(UI_PATH))

MAX_CLASSES = 150
GOLDEN_RATIO = 0.618033988749895
MIN_BOX_SIZE = 2
DEFAULT_CLASS = 1
METHOD_MANUAL = "Manual"
METHOD_GRABCUT = "GrabCut"
METHOD_MAGIC_WAND = "Magic Wand"
METHOD_AUTO_FIT = "Auto Fit"
METHODS = [
    METHOD_MANUAL,
    METHOD_GRABCUT,
    METHOD_MAGIC_WAND,
    METHOD_AUTO_FIT,
]
METHOD_HINTS = {
    METHOD_MANUAL: "Drag to draw  ·  Hover a box + Backspace to delete",
    METHOD_GRABCUT: "Drag a rough box around the object  ·  GrabCut tightens it  ·  Backspace deletes",
    METHOD_MAGIC_WAND: "Click the object (or drag a region)  ·  Choose the tolerance  ·  Backspace deletes",
    METHOD_AUTO_FIT: "Drag a rough box  ·  Auto Fit snaps it to the object  ·  Backspace deletes",
}
TOLERANCES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80]
DEFAULT_TOLERANCE = 25
MODEL_PATH = Path("runs/detect/runs/model/weights/best.pt")
MODEL_GLOB = "runs/**/weights/best.pt"
MODEL_IMAGE_SIZE = 1024
MODEL_DEVICE = None
CONFIDENCES = [0.10, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80]
DEFAULT_CONFIDENCE = 0.25
PREDICT_DUPLICATE_IOU = 0.7
WORK_MAX_SIDE = 400
GRABCUT_ITERATIONS = 4
GRABCUT_PADDING = 0.1
MIN_REGION_RATIO = 0.05
MORPH_KERNEL = np.ones((3, 3), np.uint8)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

WIDGET_TYPES = {
    "next_button": QPushButton,
    "back_button": QPushButton,
    "image_label": QLabel,
    "file_label": QLabel,
    "count_label": QLabel,
    "method_dropdown": QComboBox,
    "class_dropdown": QComboBox,
    "class_name_edit": QLineEdit,
    "image_frame": QWidget,
    "tolerance_label": QLabel,
    "tolerance_dropdown": QComboBox,
    "hint_label": QLabel,
    "model_bar": QWidget,
    "model_label": QLabel,
    "conf_dropdown": QComboBox,
    "auto_predict_check": QCheckBox,
    "predict_button": QPushButton,
}

window = None
canvas = None
ui = {}
image_files = []
image_count = 0
boxes_by_image = {}
class_names = {}
folder_key = ""
other_lines = []
saved_lines = {}
model = None
model_path = None
predicted_images = set()


@lru_cache(maxsize=None)
def GetClassColor(class_id):
    hue = (class_id * GOLDEN_RATIO) % 1.0
    saturation = (0.90, 0.65, 0.78)[class_id % 3]
    value = (1.0, 0.92)[class_id % 2]

    return QColor.fromHsvF(hue, saturation, value)


@lru_cache(maxsize=None)
def GetContrastColor(class_id):
    color = GetClassColor(class_id)

    luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()

    return QColor("#000000") if luminance > 140 else QColor("#FFFFFF")


def GetClassLabel(class_id):
    return class_names.get(class_id) or str(class_id)


def RunRefinement(function, *args):
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    try:
        return function(*args)
    except cv2.error as error:
        Verbose.Write(_TAG, f"Refinement failed: {error}", Verbose.CONSOLE_STATUS.ERROR)
        return None
    finally:
        QApplication.restoreOverrideCursor()


def ClampRect(rect, width, height):
    x0 = max(0, int(np.floor(rect.left())))
    y0 = max(0, int(np.floor(rect.top())))
    x1 = min(width, int(np.ceil(rect.right())))
    y1 = min(height, int(np.ceil(rect.bottom())))

    return x0, y0, x1, y1


def Downscale(region):
    side = max(region.shape[:2])

    if side <= WORK_MAX_SIDE:
        return region, 1.0

    scale = WORK_MAX_SIDE / side

    small = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return small, scale


def MaskToRect(mask, origin_x, origin_y, scale, min_pixels, bounds):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)

    if cv2.countNonZero(mask) < max(min_pixels, 1):
        return None

    x, y, width, height = cv2.boundingRect(mask)

    rect = QRectF(
        origin_x + x / scale, origin_y + y / scale, width / scale, height / scale
    ).intersected(bounds)

    if rect.width() < MIN_BOX_SIZE or rect.height() < MIN_BOX_SIZE:
        return None

    return rect


def GrabCut(array, box):
    height, width = array.shape[:2]
    bounds = QRectF(0, 0, width, height)

    padded = box.adjusted(
        -box.width() * GRABCUT_PADDING,
        -box.height() * GRABCUT_PADDING,
        box.width() * GRABCUT_PADDING,
        box.height() * GRABCUT_PADDING,
    )

    x0, y0, x1, y1 = ClampRect(padded, width, height)
    small, scale = Downscale(np.ascontiguousarray(array[y0:y1, x0:x1]))

    rect_x = max(int(round((box.left() - x0) * scale)), 0)
    rect_y = max(int(round((box.top() - y0) * scale)), 0)
    rect_w = min(int(round(box.width() * scale)), small.shape[1] - rect_x)
    rect_h = min(int(round(box.height() * scale)), small.shape[0] - rect_y)

    if rect_w < 4 or rect_h < 4:
        return None

    mask = np.zeros(small.shape[:2], np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        small,
        mask,
        (rect_x, rect_y, rect_w, rect_h),
        background_model,
        foreground_model,
        GRABCUT_ITERATIONS,
        cv2.GC_INIT_WITH_RECT,
    )

    foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    return MaskToRect(
        foreground, x0, y0, scale, MIN_REGION_RATIO * rect_w * rect_h, bounds
    )


def AutoFit(array, box):
    height, width = array.shape[:2]
    bounds = QRectF(0, 0, width, height)

    x0, y0, x1, y1 = ClampRect(box, width, height)

    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    small, scale = Downscale(np.ascontiguousarray(array[y0:y1, x0:x1]))

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ring = max(2, min(binary.shape) // 20)

    border = np.concatenate(
        [
            binary[:ring].ravel(),
            binary[-ring:].ravel(),
            binary[:, :ring].ravel(),
            binary[:, -ring:].ravel(),
        ]
    )

    if border.mean() > 127:
        binary = cv2.bitwise_not(binary)

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, MORPH_KERNEL)

    return MaskToRect(binary, x0, y0, scale, MIN_REGION_RATIO * binary.size, bounds)


def MagicWand(array, seed, region, tolerance):
    height, width = array.shape[:2]

    if region is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, y0, x1, y1 = ClampRect(region, width, height)

    seed_x = int(seed.x()) - x0
    seed_y = int(seed.y()) - y0

    if not (0 <= seed_x < x1 - x0 and 0 <= seed_y < y1 - y0):
        return None

    crop = cv2.GaussianBlur(np.ascontiguousarray(array[y0:y1, x0:x1]), (5, 5), 0)

    mask = np.zeros((crop.shape[0] + 2, crop.shape[1] + 2), np.uint8)
    difference = (tolerance, tolerance, tolerance)

    flags = 8 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)

    _, _, _, rect = cv2.floodFill(
        crop, mask, (seed_x, seed_y), 0, difference, difference, flags
    )

    x, y, rect_width, rect_height = rect

    if rect_width < MIN_BOX_SIZE or rect_height < MIN_BOX_SIZE:
        return None

    return QRectF(x0 + x, y0 + y, rect_width, rect_height)


REFINERS = {
    METHOD_GRABCUT: GrabCut,
    METHOD_AUTO_FIT: AutoFit,
}


class ImageCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.pixmap = QPixmap()
        self.scaled = QPixmap()
        self.array = None
        self.image_rect = QRectF()
        self.boxes = []
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.hover_index = -1

        self.label_font = QFont()
        self.label_font.setBold(True)
        self.label_font.setPointSize(9)
        self.label_metrics = QFontMetrics(self.label_font)

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def SetImage(self, image_path):
        reader = QImageReader(str(image_path))
        reader.setAutoTransform(True)
        self.pixmap = QPixmap.fromImage(reader.read())
        self.array = None
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.hover_index = -1
        self.UpdateLayoutCache()
        self.update()

    def SetBoxes(self, boxes):
        self.boxes = boxes
        self.hover_index = -1
        self.update()

    def GetArray(self):
        if self.array is None and not self.pixmap.isNull():
            image = self.pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)

            width = image.width()
            height = image.height()
            stride = image.bytesPerLine()

            buffer = np.frombuffer(
                image.constBits(), dtype=np.uint8, count=height * stride
            )

            self.array = (
                buffer.reshape(height, stride)[:, : width * 3]
                .reshape(height, width, 3)
                .copy()
            )

        return self.array

    def UpdateLayoutCache(self):
        self.scaled = QPixmap()

        if self.pixmap.isNull():
            self.image_rect = QRectF()
            return

        image_width = self.pixmap.width()
        image_height = self.pixmap.height()
        widget_width = max(self.width(), 1)
        widget_height = max(self.height(), 1)

        scale = min(widget_width / image_width, widget_height / image_height)

        width = max(1, round(image_width * scale))
        height = max(1, round(image_height * scale))

        self.image_rect = QRectF(
            (widget_width - width) // 2, (widget_height - height) // 2, width, height
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.UpdateLayoutCache()

    def WidgetToImage(self, point, clamp=False):
        rect = self.image_rect

        if rect.isEmpty():
            return None

        if clamp:
            x = min(max(point.x(), rect.left()), rect.right())
            y = min(max(point.y(), rect.top()), rect.bottom())
        elif rect.contains(point):
            x = point.x()
            y = point.y()
        else:
            return None

        return QPointF(
            (x - rect.x()) / rect.width() * self.pixmap.width(),
            (y - rect.y()) / rect.height() * self.pixmap.height(),
        )

    def ImageRectToWidget(self, rect):
        base = self.image_rect
        scale_x = base.width() / self.pixmap.width()
        scale_y = base.height() / self.pixmap.height()

        return QRectF(
            base.x() + rect.x() * scale_x,
            base.y() + rect.y() * scale_y,
            rect.width() * scale_x,
            rect.height() * scale_y,
        )

    def FindBoxAt(self, position):
        point = self.WidgetToImage(position)

        if point is None:
            return -1

        for index in range(len(self.boxes) - 1, -1, -1):
            if self.boxes[index][0].contains(point):
                return index

        return -1

    def DrawBox(self, painter, rect, color, thickness, text, text_color):
        top_color = QColor(color)
        top_color.setAlpha(120)

        bottom_color = QColor(color)
        bottom_color.setAlpha(25)

        gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0.0, top_color)
        gradient.setColorAt(1.0, bottom_color)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawRect(rect)

        painter.setPen(QPen(color, thickness))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        label_width = self.label_metrics.horizontalAdvance(text) + 14
        label_height = self.label_metrics.height() + 4

        label_y = rect.top() - label_height

        if label_y < 0:
            label_y = rect.top()

        label_rect = QRectF(rect.left(), label_y, label_width, label_height)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRect(label_rect)

        painter.setPen(text_color)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, text)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#181A20"))

        if self.pixmap.isNull():
            return

        rect = self.image_rect

        if self.scaled.isNull():
            self.scaled = self.pixmap.scaled(
                QSize(int(rect.width()), int(rect.height())),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        painter.drawPixmap(rect.topLeft(), self.scaled)
        painter.setFont(self.label_font)

        for index, (box, class_id) in enumerate(self.boxes):
            self.DrawBox(
                painter,
                self.ImageRectToWidget(box),
                GetClassColor(class_id),
                4 if index == self.hover_index else 2,
                GetClassLabel(class_id),
                GetContrastColor(class_id),
            )

        if self.drawing and self.start_point and self.end_point:
            class_id = GetSelectedClass()

            self.DrawBox(
                painter,
                self.ImageRectToWidget(
                    QRectF(self.start_point, self.end_point).normalized()
                ),
                GetClassColor(class_id),
                2,
                GetClassLabel(class_id),
                GetContrastColor(class_id),
            )

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self.pixmap.isNull():
            return

        self.setFocus()

        point = self.WidgetToImage(event.position())

        if point is None:
            return

        self.start_point = point
        self.end_point = point
        self.drawing = True
        self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = self.WidgetToImage(event.position(), True)
            self.update()
            return

        if not self.hasFocus():
            self.setFocus()

        index = self.FindBoxAt(event.position())

        if index != self.hover_index:
            self.hover_index = index
            self.update()

    def BuildBox(self):
        box = QRectF(self.start_point, self.end_point).normalized()

        dragged = box.width() >= MIN_BOX_SIZE and box.height() >= MIN_BOX_SIZE

        method = GetSelectedMethod()

        if method == METHOD_MAGIC_WAND:
            return RunRefinement(
                MagicWand,
                self.GetArray(),
                box.center() if dragged else self.start_point,
                box if dragged else None,
                GetTolerance(),
            )

        if not dragged:
            return None

        refine = REFINERS.get(method)

        if refine is None:
            return box

        result = RunRefinement(refine, self.GetArray(), box)

        return box if result is None else result

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self.drawing:
            return

        end = self.WidgetToImage(event.position(), True)

        if end is not None:
            self.end_point = end

        if self.start_point and self.end_point:
            box = self.BuildBox()

            if box is not None:
                self.boxes.append((box, GetSelectedClass()))

        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.hover_index = self.FindBoxAt(event.position())

        OnBoxesChanged()
        self.update()

    def leaveEvent(self, event):
        if self.hover_index != -1:
            self.hover_index = -1
            self.update()

        super().leaveEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_P and model_path is not None and not self.drawing:
            PredictCurrent()
            return

        if (
            event.key() == Qt.Key.Key_Backspace
            and not self.drawing
            and 0 <= self.hover_index < len(self.boxes)
        ):
            del self.boxes[self.hover_index]
            self.hover_index = -1

            OnBoxesChanged()
            self.update()
            return

        super().keyPressEvent(event)


def Start(folder):
    global window

    if not UI_FILE.open(QFile.OpenModeFlag.ReadOnly):
        Verbose.Write(
            _TAG, "Error to initialize interface", Verbose.CONSOLE_STATUS.ERROR
        )
        return 1

    window = QUiLoader().load(UI_FILE)
    UI_FILE.close()

    if window is None:
        Verbose.Write(_TAG, "Error to load UI file", Verbose.CONSOLE_STATUS.ERROR)
        return 1

    GetImages(folder)

    if not image_files:
        Verbose.Write(_TAG, "No images found", Verbose.CONSOLE_STATUS.ERROR)
        return 1

    LoadMetadata(folder)

    if not Configure():
        return 1

    window.show()

    return APP.exec()


def BindWidgets():
    for name, kind in WIDGET_TYPES.items():
        widget = window.findChild(kind, name)

        if widget is None:
            Verbose.Write(_TAG, f"{name} not found", Verbose.CONSOLE_STATUS.ERROR)
            return False

        ui[name] = widget

    return True


def Configure():
    global canvas

    if not BindWidgets():
        return False

    ui["next_button"].clicked.connect(Next)
    ui["back_button"].clicked.connect(Back)

    frame = ui["image_frame"]
    layout = frame.layout()

    if layout is None:
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)

    canvas = ImageCanvas(frame)
    canvas.setObjectName("image_canvas")
    canvas.setCursor(Qt.CursorShape.CrossCursor)
    layout.addWidget(canvas)

    for tolerance in TOLERANCES:
        ui["tolerance_dropdown"].addItem(str(tolerance), tolerance)

    ui["tolerance_dropdown"].setCurrentIndex(TOLERANCES.index(DEFAULT_TOLERANCE))

    ui["method_dropdown"].addItems(METHODS)
    ui["method_dropdown"].currentIndexChanged.connect(OnMethodChanged)
    OnMethodChanged()

    PopulateClasses()

    ui["class_dropdown"].currentIndexChanged.connect(SyncClassNameEdit)
    ui["class_name_edit"].editingFinished.connect(RenameSelectedClass)
    ui["class_name_edit"].returnPressed.connect(canvas.setFocus)

    ConfigureModel()
    LoadCurrentImage()

    return True


def LoadCurrentImage():
    path = image_files[image_count]

    canvas.SetImage(path)
    canvas.SetBoxes(boxes_by_image.setdefault(path, []))

    if canvas.pixmap.isNull():
        Verbose.Write(
            _TAG, f"Failed to load image: {path}", Verbose.CONSOLE_STATUS.ERROR
        )

    UpdateImageLabelText()
    UpdateNavigation()
    UpdateStatus()
    AutoPredictIfNeeded()


def Navigate(step):
    global image_count

    target = image_count + step

    if 0 <= target < len(image_files):
        image_count = target
        LoadCurrentImage()


def Next():
    Navigate(1)


def Back():
    Navigate(-1)


def NormalizeFolder(text):
    return os.path.normcase(os.path.normpath(text.strip()))


def ReadMetadataLines():
    return [str(line).rstrip("\r\n") for line in Metadata.Read()]


def WriteMetadataLines(lines):
    if not Metadata.Rewrite(lines):
        raise OSError("metadata file is not open")


def SerializeBoxes(index, boxes):
    name = image_files[index].name
    lines = []

    for box, class_id in boxes:
        start = (round(box.left()), round(box.top()))
        end = (round(box.right()), round(box.bottom()))

        lines.append(f"{folder_key}|{index}|{(start, end)}|{class_id}|{name}")

    return lines


def ParseMetadataLine(parts, name_to_index):
    try:
        (x1, y1), (x2, y2) = ast.literal_eval(parts[2])
        class_id = int(parts[3]) if len(parts) > 3 else DEFAULT_CLASS

        if len(parts) > 4 and parts[4]:
            index = name_to_index.get(parts[4])
        else:
            index = int(parts[1])

        if index is None or not 0 <= index < len(image_files):
            return None

        box = QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized()

        return index, box, class_id
    except (ValueError, TypeError, SyntaxError, IndexError):
        return None


def LoadMetadata(folder):
    global folder_key, image_count

    folder_key = str(folder)
    key = NormalizeFolder(folder_key)
    name_to_index = {file.name: i for i, file in enumerate(image_files)}

    other_lines.clear()
    saved_lines.clear()

    try:
        lines = ReadMetadataLines()
    except OSError:
        lines = []

    seen = set()

    for line in lines:
        if not line.strip():
            continue

        parts = line.split("|")

        if NormalizeFolder(parts[0]) != key:
            other_lines.append(line)
            continue

        parsed = ParseMetadataLine(parts, name_to_index)

        if parsed is None:
            other_lines.append(line)
            Verbose.Write(
                _TAG, f"Ignored metadata line: {line}", Verbose.CONSOLE_STATUS.ERROR
            )
            continue

        index, box, class_id = parsed
        signature = (
            index,
            round(box.left()),
            round(box.top()),
            round(box.right()),
            round(box.bottom()),
            class_id,
        )

        if signature in seen:
            continue

        seen.add(signature)
        boxes_by_image.setdefault(image_files[index], []).append((box, class_id))

    for index, path in enumerate(image_files):
        boxes = boxes_by_image.get(path)

        if boxes:
            saved_lines[index] = SerializeBoxes(index, boxes)

    if saved_lines:
        image_count = max(saved_lines)


def SaveMetadata():
    lines = list(other_lines)

    for index in sorted(saved_lines):
        lines.extend(saved_lines[index])

    try:
        WriteMetadataLines(lines)
    except OSError as error:
        Verbose.Write(
            _TAG, f"Failed to save metadata: {error}", Verbose.CONSOLE_STATUS.ERROR
        )


def OnBoxesChanged():
    lines = SerializeBoxes(image_count, canvas.boxes)

    if lines:
        saved_lines[image_count] = lines
    else:
        saved_lines.pop(image_count, None)

    SaveMetadata()
    UpdateStatus()


def UpdateNavigation():
    ui["back_button"].setEnabled(image_count > 0)
    ui["next_button"].setEnabled(image_count < len(image_files) - 1)


def UpdateStatus():
    total = len(canvas.boxes)

    ui["file_label"].setText(image_files[image_count].name)
    ui["count_label"].setText(f"{total} box" if total == 1 else f"{total} boxes")


def FormatClassItem(class_id):
    name = class_names.get(class_id)

    return f"{class_id}  ·  {name}" if name else str(class_id)


def PopulateClasses():
    dropdown = ui["class_dropdown"]

    dropdown.blockSignals(True)
    dropdown.clear()

    for class_id in range(1, MAX_CLASSES + 1):
        dropdown.addItem(FormatClassItem(class_id), class_id)

    dropdown.setCurrentIndex(0)
    dropdown.blockSignals(False)

    SyncClassNameEdit()


def GetSelectedClass():
    class_id = ui["class_dropdown"].currentData()

    return class_id if class_id is not None else 0


def SyncClassNameEdit():
    ui["class_name_edit"].setText(class_names.get(GetSelectedClass(), ""))


def RenameSelectedClass():
    class_id = GetSelectedClass()
    name = ui["class_name_edit"].text().strip()

    if name:
        class_names[class_id] = name
    else:
        class_names.pop(class_id, None)

    ui["class_dropdown"].setItemText(
        ui["class_dropdown"].currentIndex(), FormatClassItem(class_id)
    )

    canvas.update()


def GetSelectedMethod():
    return ui["method_dropdown"].currentText()


def FindModelPath():
    if MODEL_PATH.exists():
        return MODEL_PATH

    found = [path for path in Path(".").glob(MODEL_GLOB) if path.is_file()]

    if not found:
        return None

    return max(found, key=lambda path: path.stat().st_mtime)


def ConfigureModel():
    global model_path

    model_path = FindModelPath()

    if model_path is None:
        ui["model_bar"].setVisible(False)
        return

    for confidence in CONFIDENCES:
        ui["conf_dropdown"].addItem(f"{confidence:.2f}", confidence)

    ui["conf_dropdown"].setCurrentIndex(CONFIDENCES.index(DEFAULT_CONFIDENCE))
    ui["model_label"].setText(
        f"Model  ·  {model_path.parent.parent.name} / {model_path.name}"
    )
    ui["predict_button"].clicked.connect(PredictCurrent)
    ui["model_bar"].setVisible(True)


def DisableModel():
    global model_path

    model_path = None
    ui["model_bar"].setVisible(False)


def GetConfidence():
    confidence = ui["conf_dropdown"].currentData()

    return confidence if confidence is not None else DEFAULT_CONFIDENCE


def RefreshClassItems():
    dropdown = ui["class_dropdown"]

    for index in range(dropdown.count()):
        dropdown.setItemText(index, FormatClassItem(dropdown.itemData(index)))

    SyncClassNameEdit()


def ApplyModelNames(names):
    items = names.items() if isinstance(names, dict) else enumerate(names)

    for index, name in items:
        class_names.setdefault(int(index) + 1, str(name))

    RefreshClassItems()


def LoadModel():
    global model

    if model is None:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        ApplyModelNames(model.names)

    return model


def PredictBoxes(path, confidence):
    result = LoadModel().predict(
        source=str(path),
        conf=confidence,
        imgsz=MODEL_IMAGE_SIZE,
        device=MODEL_DEVICE,
        verbose=False,
    )[0]

    boxes = []

    for corners, class_index in zip(
        result.boxes.xyxy.tolist(), result.boxes.cls.tolist()
    ):
        x1, y1, x2, y2 = corners

        boxes.append(
            (
                QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized(),
                int(class_index) + 1,
            )
        )

    return boxes


def BoxIou(first, second):
    overlap = first.intersected(second)

    if overlap.isEmpty():
        return 0.0

    area = overlap.width() * overlap.height()
    union = first.width() * first.height() + second.width() * second.height() - area

    return area / union if union > 0 else 0.0


def PredictCurrent():
    if model_path is None or canvas is None:
        return

    path = image_files[image_count]

    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    try:
        predictions = PredictBoxes(path, GetConfidence())
    except ImportError:
        Verbose.Write(
            _TAG,
            "Error: package ultralytics is not installed.",
            Verbose.CONSOLE_STATUS.ERROR,
        )
        DisableModel()
        return
    except Exception as error:
        Verbose.Write(_TAG, f"Prediction failed: {error}", Verbose.CONSOLE_STATUS.ERROR)
        return
    finally:
        QApplication.restoreOverrideCursor()

    predicted_images.add(path)
    added = 0

    for box, class_id in predictions:
        if any(
            existing_class == class_id and BoxIou(box, existing) > PREDICT_DUPLICATE_IOU
            for existing, existing_class in canvas.boxes
        ):
            continue

        canvas.boxes.append((box, class_id))
        added += 1

    if added:
        canvas.update()
        OnBoxesChanged()

    UpdateStatus()

    suffix = f"+{added} predicted" if added else "nothing new detected"
    ui["count_label"].setText(f"{ui['count_label'].text()}  ·  {suffix}")


def AutoPredictIfNeeded():
    if model_path is None or not ui["auto_predict_check"].isChecked():
        return

    if image_files[image_count] in predicted_images or canvas.boxes:
        return

    PredictCurrent()


def GetTolerance():
    tolerance = ui["tolerance_dropdown"].currentData()

    return tolerance if tolerance is not None else DEFAULT_TOLERANCE


def OnMethodChanged():
    method = GetSelectedMethod()
    uses_tolerance = method == METHOD_MAGIC_WAND

    ui["tolerance_label"].setVisible(uses_tolerance)
    ui["tolerance_dropdown"].setVisible(uses_tolerance)
    ui["hint_label"].setText(METHOD_HINTS.get(method, ""))


def SetLabelText(label_name, text):
    label = ui.get(label_name)

    if label is not None:
        label.setText(text)


def UpdateImageLabelText():
    SetLabelText("image_label", f"{image_count + 1} / {len(image_files)}")


def GetImages(folder):
    image_files.clear()

    for file in Folder.GetFolder(folder).iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file)

    image_files.sort()
