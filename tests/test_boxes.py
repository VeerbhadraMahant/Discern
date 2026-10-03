import pytest

from discern.vision.boxes import Box, clip, expand, iou, scale, union_box


def test_iou_identical_disjoint_and_partial() -> None:
    assert iou(Box(0, 0, 10, 10), Box(0, 0, 10, 10)) == 1.0
    assert iou(Box(0, 0, 10, 10), Box(20, 20, 30, 30)) == 0.0
    # overlap 5x10 = 50, union 100 + 100 - 50 = 150
    assert iou(Box(0, 0, 10, 10), Box(5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_iou_touching_edges_and_degenerate() -> None:
    assert iou(Box(0, 0, 10, 10), Box(10, 0, 20, 10)) == 0.0
    assert iou(Box(0, 0, 0, 0), Box(0, 0, 0, 0)) == 0.0


def test_expand_by_alpha() -> None:
    assert expand(Box(10, 10, 30, 50), 0.25) == Box(5, 0, 35, 60)


def test_clip_to_frame() -> None:
    assert clip(Box(-5, -5, 120, 80), 100, 60) == Box(0, 0, 100, 60)


def test_scale_between_spaces() -> None:
    assert scale(Box(10, 20, 30, 40), 2.0) == Box(20, 40, 60, 80)
    assert scale(Box(10, 20, 30, 40), 2.0, 0.5) == Box(20, 10, 60, 20)


def test_union_box() -> None:
    assert union_box([Box(0, 0, 10, 10), Box(5, -5, 20, 8)]) == Box(0, -5, 20, 10)
    with pytest.raises(ValueError):
        union_box([])
