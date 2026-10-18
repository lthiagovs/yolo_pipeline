from pathlib import Path

from ultralytics import YOLO

import src.helpers.verbose as Verbose

_TAG = "MODEL"

MODEL_PATH = Path("runs/detect/runs/model/weights/best.pt")
OUTPUT_FOLDER = Path("runs/detect/predict")

_model = None


def LoadModel():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            Verbose.Write(_TAG, "Error: Model not found.", Verbose.CONSOLE_STATUS.ERROR)
            return None
        _model = YOLO(str(MODEL_PATH))
    return _model


def DetectImage(path, conf=0.25, imgsz=1024):
    model = LoadModel()
    if model is None:
        return

    if not Path(path).is_file():
        Verbose.Write(_TAG, "Error: Image not found.", Verbose.CONSOLE_STATUS.ERROR)
        return

    results = model.predict(
        source=str(path),
        conf=conf,
        imgsz=imgsz,
        device="cpu",
        verbose=False,
    )
    r = results[0]

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_FOLDER / f"{Path(path).stem}_detected.jpg"
    r.save(filename=str(out))

    Verbose.Write(_TAG, f"Detections: {len(r.boxes)}", Verbose.CONSOLE_STATUS.INFO)
    Verbose.Write(_TAG, "Saved on:" + out, Verbose.CONSOLE_STATUS.SUCCESS)


def DetectWebcam(conf=0.25, imgsz=1024):
    model = LoadModel()
    if model is None:
        return
    model.predict(source=0, show=True, conf=conf, imgsz=imgsz, device="cpu")
