from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from discern.agent.schemas import ShotPlan
from discern.eval.ablation import (
    decision_stats,
    delta_table,
    detection_cache_name,
    pair_frequency,
    plan_cache_name,
    rescale_detections,
)
from discern.eval.runner import detect_all
from discern.eval.types import AnnotatedImage
from discern.models.roles import Detection
from discern.vision.boxes import Box


def det(box: Box) -> Detection:
    return Detection(box=box, label="car", score=0.9, detector="t")


def test_rescale_divides_by_factor_and_clips_to_frame() -> None:
    out = rescale_detections([det(Box(20, 40, 400, 200))], 2, 100, 80)
    assert out[0].box == Box(10, 20, 100, 80)
    assert rescale_detections([det(Box(-5, 1, 50, 60))], 1, 40, 40)[0].box == Box(0, 1, 40, 40)


def test_cache_name_separates_arm_detector_dataset_revision_vlm_and_memory() -> None:
    base = ("detas_x", "owlv2-base", "darkface", "a" * 40, "vlm1", "v1")
    names = {
        detection_cache_name(*base),
        detection_cache_name("detas", *base[1:]),
        detection_cache_name(base[0], "yolo-world-v2", *base[2:]),
        detection_cache_name(*base[:2], "bdd100k_night", *base[3:]),
        detection_cache_name(*base[:3], "b" * 40, *base[4:]),
        detection_cache_name(*base[:4], "vlm2", base[5]),
        detection_cache_name(*base[:5], "v2"),
    }
    assert len(names) == 7
    assert detection_cache_name("detas_xp", *base[1:]) not in names
    assert "v1" in detection_cache_name(*base) and "a" * 10 in detection_cache_name(*base)
    assert plan_cache_name("darkface", "vlm1", "v1") != plan_cache_name("darkface", "vlm1", "v2")
    assert plan_cache_name("darkface", "vlm1", "v1") == "plans-darkface-vlm1-v1.json"  # as before
    xp = plan_cache_name("darkface", "vlm1", "v1", "detas_xp")
    assert xp != plan_cache_name("darkface", "vlm1", "v1") and "detas_xp" in xp


def test_decision_stats_rates_and_accuracy() -> None:
    plans = [
        ShotPlan(restorer="dehaze", use_restored=True, sr_factor=2),
        ShotPlan(restorer="dehaze", use_restored=False, sr_factor=None),
        ShotPlan(restorer="none", use_restored=False, sr_factor=None),
        ShotPlan(restorer="none", use_restored=False, sr_factor=4),
    ]
    stats = decision_stats(plans, ["fog", "fog", "normal", "rain"], ["fog", "rain", "normal", None])
    assert stats == {
        "restore_rate": 0.5,
        "accept_rate": 0.25,
        "sr_rate": 0.5,
        "perception_acc": 0.5,
    }
    assert decision_stats([], [], [])["sr_rate"] == 0.0


def test_pair_frequency_ignores_order() -> None:
    assert pair_frequency([["a", "b"], ["b", "a"], ["a", "c"]]) == {"a+b": 2, "a+c": 1}


def test_delta_table_rows_deltas_mean_and_missing() -> None:
    f1 = {
        ("d1", "detas", "e2e_k2"): 0.40,
        ("d1", "detas_x", "e2e_k2"): 0.45,
        ("d2", "detas", "e2e_k2"): 0.50,
        ("d2", "detas_x", "e2e_k2"): 0.48,
        ("d1", "detas", "med_k2"): 0.30,
    }
    table = delta_table(f1, ["d1", "d2"], ["e2e_k2", "med_k2"], ["detas", "detas_x"])
    assert "| d1 | e2e_k2 | 0.400 | 0.450 | +0.050 |" in table
    assert "| d2 | e2e_k2 | 0.500 | 0.480 | -0.020 |" in table
    assert "| mean | e2e_k2 | 0.450 | 0.465 | +0.015 |" in table
    assert "| d1 | med_k2 | 0.300 | n/a | n/a |" in table
    assert "mean | med_k2" not in table


def test_delta_table_has_a_column_and_delta_per_arm_against_detas() -> None:
    f1 = {
        ("d1", "detas", "m"): 0.40,
        ("d1", "detas_x", "m"): 0.45,
        ("d1", "detas_xp", "m"): 0.50,
        ("d2", "detas", "m"): 0.50,
        ("d2", "detas_xp", "m"): 0.44,
    }
    table = delta_table(f1, ["d1", "d2"], ["m"])
    assert "| dataset | metric | DetAS | DetAS-X | delta | DetAS-XP | delta |" in table
    assert "| d1 | m | 0.400 | 0.450 | +0.050 | 0.500 | +0.100 |" in table
    assert "| d2 | m | 0.500 | n/a | n/a | 0.440 | -0.060 |" in table
    assert "| mean | m | 0.450 | 0.450 | +0.050 | 0.470 | +0.020 |" in table


def test_detect_all_postprocess_maps_back_and_is_cached(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    PILImage.fromarray(np.zeros((10, 20, 3), dtype=np.uint8)).save(path)
    image = AnnotatedImage(image_id="a", path=path, width=20, height=10, objects=())
    seen: list[tuple[int, ...]] = []

    class Doubling:
        name = "d"

        def detect(self, img: np.ndarray, targets: list[str]) -> list[Detection]:  # type: ignore[override]
            seen.append(img.shape[:2])
            return [det(Box(0, 0, 40, 20))]

    cache = tmp_path / "c.json"
    kwargs = {
        "preprocess": lambda a, rgb: np.repeat(np.repeat(rgb, 2, 0), 2, 1),
        "postprocess": lambda a, rgb, dets: rescale_detections(
            dets, 2, rgb.shape[1], rgb.shape[0]
        ),
    }
    out = detect_all(Doubling(), [image], ["car"], cache, **kwargs)  # type: ignore[arg-type]
    assert seen == [(20, 40)] and out["a"][0].box == Box(0, 0, 20, 10)
    again = detect_all(Doubling(), [image], ["car"], cache, **kwargs)  # type: ignore[arg-type]
    assert seen == [(20, 40)] and again == out
