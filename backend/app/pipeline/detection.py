import uuid

import numpy as np
from ultralytics import YOLO

from app.config import YOLO_CONFIDENCE, YOLO_MODEL_PATH
from app.schemas import BoundingBox, Detection

_model: YOLO | None = None

# COCO classes this pure-detection app cares about: people plus the vehicle
# classes needed for street/traffic scenes (bicycle, car, motorcycle, bus, truck).
DETECTABLE_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(YOLO_MODEL_PATH)
    return _model


def person_detection(img: np.ndarray) -> list[Detection]:
    """Detect people and vehicles (YOLOv8n/COCO) in the (restored) frame."""
    model = _get_model()
    results = model.predict(img, classes=list(DETECTABLE_CLASSES.keys()), conf=YOLO_CONFIDENCE, verbose=False)[0]

    detections: list[Detection] = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = DETECTABLE_CLASSES.get(int(box.cls[0]), "object")
        detections.append(
            Detection(
                id=f"{label}_{uuid.uuid4().hex[:8]}",
                label=label,
                confidence=round(float(box.conf[0]), 3),
                box=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                source="person_detection",
            )
        )
    return detections


def people_only(detections: list[Detection]) -> list[Detection]:
    """Subset used for PPE/zone violation reasoning, which only makes sense for people."""
    return [d for d in detections if d.label == "person"]
