import pytest

from discern.eval.metrics import Counts, match_image
from discern.eval.types import GroundTruthBox
from discern.models.roles import Detection
from discern.vision.boxes import Box


def det(box: Box, label: str = "car", score: float = 0.9) -> Detection:
    return Detection(box=box, label=label, score=score, detector="t")


def gt(box: Box, label: str = "car") -> GroundTruthBox:
    return GroundTruthBox(box=box, label=label)


A = Box(0, 0, 10, 10)
B = Box(50, 50, 60, 60)


def test_perfect_match() -> None:
    c = match_image([det(A), det(B)], [gt(A), gt(B)])
    assert c == Counts(tp=2, fp=0, fn=0)
    assert c.f1 == 1.0


def test_false_positive_and_false_negative() -> None:
    c = match_image([det(A), det(Box(100, 100, 110, 110))], [gt(A), gt(B)])
    assert c == Counts(tp=1, fp=1, fn=1)
    assert c.f1 == pytest.approx(0.5)


def test_class_must_match() -> None:
    assert match_image([det(A, "person")], [gt(A, "car")]) == Counts(tp=0, fp=1, fn=1)


def test_iou_threshold_is_inclusive_at_half() -> None:
    # IoU of (0,0,10,10) and (5,0,15,10) is 1/3: below 0.5.
    assert match_image([det(Box(5, 0, 15, 10))], [gt(A)]).tp == 0
    # (0,0,10,10) vs (0,0,10,5): IoU exactly 0.5 counts.
    assert match_image([det(Box(0, 0, 10, 5))], [gt(A)]).tp == 1


def test_each_ground_truth_matches_at_most_once() -> None:
    c = match_image([det(A, score=0.9), det(A, score=0.8)], [gt(A)])
    assert c == Counts(tp=1, fp=1, fn=0)


def test_higher_score_claims_the_ground_truth_first() -> None:
    near = Box(0, 0, 10, 6)  # IoU 0.6 with A
    exact = Box(0, 0, 10, 10)  # IoU 1.0 with A
    c = match_image([det(near, score=0.95), det(exact, score=0.5)], [gt(A)])
    assert c == Counts(tp=1, fp=1, fn=0)  # greedy by score, not by best overlap


def test_counts_add_and_empty_cases() -> None:
    total = Counts(1, 1, 0) + Counts(2, 0, 3)
    assert total == Counts(3, 1, 3)
    assert Counts().f1 == 0.0
    assert match_image([], []) == Counts()
