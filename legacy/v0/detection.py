"""v0 baseline detector: YOLOv8n on COCO classes. Frozen; used only as an evaluation baseline."""
import numpy as np
from ultralytics import YOLO

YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE = 0.25

# COCO classes v0 cared about: people plus the vehicle classes.
DETECTABLE_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(YOLO_MODEL_PATH)
    return _model


def detect(img: np.ndarray) -> list[dict]:
    """Return detections as {label, confidence, xyxy} for a BGR frame."""
    results = _get_model().predict(
        img, classes=list(DETECTABLE_CLASSES), conf=YOLO_CONFIDENCE, verbose=False
    )[0]
    return [
        {
            "label": DETECTABLE_CLASSES.get(int(box.cls[0]), "object"),
            "confidence": round(float(box.conf[0]), 3),
            "xyxy": box.xyxy[0].tolist(),
        }
        for box in results.boxes
    ]
