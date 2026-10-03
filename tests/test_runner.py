from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from discern.eval.runner import detect_all, evaluate, tune_threshold
from discern.eval.types import AnnotatedImage, GroundTruthBox
from discern.models.roles import Detection
from discern.vision.boxes import Box

GT = Box(0, 0, 10, 10)


def det(box: Box, score: float) -> Detection:
    return Detection(box=box, label="car", score=score, detector="t")


def image(tmp_path: Path, name: str) -> AnnotatedImage:
    path = tmp_path / f"{name}.png"
    PILImage.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(path)
    return AnnotatedImage(
        image_id=name,
        path=path,
        width=20,
        height=20,
        objects=(GroundTruthBox(box=GT, label="car"),),
    )


def test_tune_threshold_drops_low_score_false_positives(tmp_path: Path) -> None:
    images = [image(tmp_path, "a"), image(tmp_path, "b")]
    dets = {
        "a": [det(GT, 0.9), det(Box(15, 15, 19, 19), 0.2)],
        "b": [det(GT, 0.8), det(Box(12, 12, 18, 18), 0.3)],
    }
    t = tune_threshold(dets, images)
    assert 0.3 < t <= 0.8
    assert evaluate(dets, images, t).f1 == 1.0
    assert evaluate(dets, images, 0.05).f1 < 1.0


class CountingDetector:
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: np.ndarray, targets: list[str]) -> list[Detection]:  # type: ignore[override]
        self.calls += 1
        return [det(GT, 0.9)]


def test_detect_all_caches_to_disk(tmp_path: Path) -> None:
    images = [image(tmp_path, "a"), image(tmp_path, "b")]
    cache = tmp_path / "cache" / "d.json"
    first = CountingDetector()
    out = detect_all(first, images, ["car"], cache)
    assert first.calls == 2 and out["a"][0].box == GT
    second = CountingDetector()
    assert detect_all(second, images, ["car"], cache) == out
    assert second.calls == 0
