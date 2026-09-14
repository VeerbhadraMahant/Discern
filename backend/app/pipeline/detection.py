import uuid

import numpy as np
from ultralytics import YOLO

from app.config import YOLO_CONFIDENCE, YOLO_MODEL_PATH
from app.schemas import BoundingBox, Detection

_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(YOLO_MODEL_PATH)
    return _model


def person_detection(img: np.ndarray) -> list[Detection]:
    model = _get_model()
    results = model.predict(img, classes=[0], conf=YOLO_CONFIDENCE, verbose=False)[0]

    detections: list[Detection] = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            Detection(
                id=f"person_{uuid.uuid4().hex[:8]}",
                label="person",
                confidence=round(float(box.conf[0]), 3),
                box=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                source="person_detection",
            )
        )
    return detections
