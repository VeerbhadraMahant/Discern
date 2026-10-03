"""Detector evaluation: cached raw detections, threshold tuned on the harvest split,
F1@0.5 reported on the gate split, runs logged to MLflow."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from pydantic import TypeAdapter

from discern.eval.datasets import DATA_DIR
from discern.eval.metrics import Counts, match_image
from discern.eval.types import AnnotatedImage
from discern.models.roles import SCORE_FLOOR, Detection, Detector

THRESHOLD_GRID = [round(t, 2) for t in np.arange(SCORE_FLOOR, 0.951, 0.05)]
_DETECTIONS = TypeAdapter(list[Detection])

DetectionMap = Mapping[str, Sequence[Detection]]  # image_id -> raw detections


def dataset_targets(images: Sequence[AnnotatedImage]) -> list[str]:
    """Labels prompted to open-vocabulary detectors: those present in the dataset's ground truth."""
    return sorted({o.label for a in images for o in a.objects})


def load_rgb(path: Path) -> np.ndarray:
    with PILImage.open(path) as im:
        return np.asarray(im.convert("RGB"))


def detect_all(
    detector: Detector,
    images: Sequence[AnnotatedImage],
    targets: Sequence[str],
    cache_file: Path | None = None,
    preprocess: Callable[[AnnotatedImage, np.ndarray], np.ndarray] | None = None,
) -> dict[str, list[Detection]]:
    """Run `detector` on every image, reusing results cached in `cache_file`."""
    cached: dict[str, list[Detection]] = {}
    if cache_file is not None and cache_file.exists():
        raw = json.loads(cache_file.read_text())
        cached = {k: _DETECTIONS.validate_python(v) for k, v in raw.items()}
    for a in images:
        if a.image_id in cached:
            continue
        img = load_rgb(a.path)
        if preprocess is not None:
            img = preprocess(a, img)
        cached[a.image_id] = detector.detect(img, targets)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(
            json.dumps(
                {k: _DETECTIONS.dump_python(v, mode="json") for k, v in cached.items()}
            ).encode()
        )
    return {a.image_id: cached[a.image_id] for a in images}


def evaluate(
    detections: DetectionMap, images: Sequence[AnnotatedImage], threshold: float
) -> Counts:
    total = Counts()
    for a in images:
        kept = [d for d in detections[a.image_id] if d.score >= threshold]
        total += match_image(kept, a.objects)
    return total


def tune_threshold(detections: DetectionMap, images: Sequence[AnnotatedImage]) -> float:
    """Score threshold with the best micro-F1 on `images` (ties go to the higher threshold)."""
    best_t, best_f1 = THRESHOLD_GRID[0], -1.0
    for t in THRESHOLD_GRID:
        f1 = evaluate(detections, images, t).f1
        if f1 >= best_f1:
            best_t, best_f1 = t, f1
    return best_t


@dataclass(frozen=True)
class EvalResult:
    method: str
    dataset: str
    threshold: float
    counts: Counts

    @property
    def f1(self) -> float:
        return self.counts.f1


def score_method(
    method: str,
    dataset: str,
    gate: Sequence[AnnotatedImage],
    gate_detections: DetectionMap,
    harvest: Sequence[AnnotatedImage],
    harvest_detections: DetectionMap,
) -> EvalResult:
    threshold = tune_threshold(harvest_detections, harvest)
    return EvalResult(method, dataset, threshold, evaluate(gate_detections, gate, threshold))


def cache_path(method: str, dataset: str, revision: str) -> Path:
    return DATA_DIR / "_cache" / method / f"{dataset}-{revision[:10]}.json"
