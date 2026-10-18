from pathlib import Path

from ultralytics import YOLO

import verbose as Verbose

_TAG = "YOLO"

DATASET = Path("dataset/data.yaml")
MODEL = "yolo11s.pt"
EPOCHS = 200
IMAGE_SIZE = 1024
BATCH_SIZE = 8


def Train():
    if not DATASET.exists():
        Verbose.Write(_TAG, "Error: Dataset not found", Verbose.CONSOLE_STATUS.ERROR)

        return

    model = YOLO(MODEL)

    model.train(
        data=str(DATASET),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project="runs",
        name="capacitor_detector",
        patience=30,
        workers=1,
        cache=True,
        cos_lr=True,
        close_mosaic=15,
        flipud=0.5,
        fliplr=0.5,
        degrees=90,
        scale=0.3,
        mosaic=1.0,
        mixup=0.0,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.4,
    )

    print(f"Modelo salvo em: {model.trainer.save_dir}/weights/best.pt")
