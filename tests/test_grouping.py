import numpy as np

from discern.config.settings import GroupingThresholds, load_settings
from discern.models.roles import Detection, Image
from discern.vision.boxes import Box
from discern.vision.grouping import group_detections

THRESHOLDS = load_settings().thresholds.grouping


def det(box: tuple[float, float, float, float], score: float, detector: str = "a") -> Detection:
    return Detection(box=Box(*box), label="thing", score=score, detector=detector)


def blob_image() -> Image:
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img[30:70, 40:80] = (230, 40, 40)  # one bright object
    return img


def test_overlapping_boxes_from_two_detectors_merge() -> None:
    a = det((40, 30, 80, 70), 0.9, "a")
    b = det((42, 32, 82, 72), 0.6, "b")
    (group,) = group_detections(blob_image(), [b, a], THRESHOLDS)
    assert group.anchor == a
    assert group.members == (a, b)
    assert group.box == a.box


def test_shifted_boxes_below_iou_threshold_stay_separate() -> None:
    a = det((40, 30, 80, 70), 0.9)
    b = det((60, 30, 100, 70), 0.8)  # IoU 1/3
    groups = group_detections(blob_image(), [a, b], THRESHOLDS)
    assert [g.anchor for g in groups] == [a, b]


def split_image() -> Image:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:60] = (255, 0, 0)
    img[60:] = (0, 0, 255)
    return img


def test_overlapping_boxes_with_dissimilar_crops_stay_separate() -> None:
    a = det((0, 0, 100, 60), 0.9)  # IoU 0.6 with b, crop cosine about 0.8
    b = det((0, 0, 100, 100), 0.8)
    assert len(group_detections(split_image(), [a, b], THRESHOLDS)) == 1
    strict = GroupingThresholds(**{**THRESHOLDS.model_dump(), "phi_vis": 0.95})
    assert len(group_detections(split_image(), [a, b], strict)) == 2


def test_dense_row_does_not_chain_merge() -> None:
    img = np.full((60, 200, 3), 120, dtype=np.uint8)
    # Neighbours overlap with IoU 0.6, two apart 0.33; chaining would collapse the row.
    dets = [det((10.0 * i, 10, 10.0 * i + 40, 50), 0.9 - 0.05 * i) for i in range(6)]
    groups = group_detections(img, dets, THRESHOLDS)
    assert [len(g.members) for g in groups] == [2, 2, 2]
    assert [g.anchor for g in groups] == [dets[0], dets[2], dets[4]]


def test_empty_input() -> None:
    assert group_detections(blob_image(), [], THRESHOLDS) == []


def test_deterministic() -> None:
    dets = [det((40, 30, 80, 70), 0.9), det((41, 31, 81, 71), 0.9), det((120, 30, 160, 70), 0.5)]
    assert group_detections(blob_image(), dets, THRESHOLDS) == group_detections(
        blob_image(), dets, THRESHOLDS
    )


def test_box_outside_image_does_not_crash() -> None:
    groups = group_detections(blob_image(), [det((290, 90, 400, 200), 0.5)], THRESHOLDS)
    assert len(groups) == 1
