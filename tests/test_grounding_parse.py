from discern.models.grounding_parse import parse_grounding

TARGETS = ["car", "person"]


def _parse(text: str) -> list:
    return parse_grounding(text, 2000, 1000, TARGETS, "vlm")


def test_valid_converts_relative_to_pixels() -> None:
    dets = _parse('[{"bbox_2d": [100, 200, 500, 800], "label": "car"}]')
    assert len(dets) == 1
    assert tuple(dets[0].box) == (200.0, 200.0, 1000.0, 800.0)
    assert dets[0].label == "car" and dets[0].score == 1.0 and dets[0].detector == "vlm"


def test_fenced_json() -> None:
    dets = _parse('Here:\n```json\n[{"bbox_2d": [0, 0, 500, 500], "label": "person"}]\n```')
    assert [d.label for d in dets] == ["person"]


def test_malformed_returns_empty() -> None:
    assert _parse('[{"bbox_2d": [1, 2, 3') == []
    assert _parse("no objects") == []
    assert _parse('{"bbox_2d": [0, 0, 5, 5], "label": "car"}') == []


def test_unknown_label_dropped_and_case_normalised() -> None:
    dets = _parse(
        '[{"bbox_2d": [0, 0, 500, 500], "label": "dog"},'
        ' {"bbox_2d": [0, 0, 500, 500], "label": "Car"}]'
    )
    assert [d.label for d in dets] == ["car"]


def test_out_of_range_clipped_and_degenerate_dropped() -> None:
    dets = _parse(
        '[{"bbox_2d": [-50, 900, 1200, 1100], "label": "car"},'
        ' {"bbox_2d": [10, 10, 10, 90], "label": "car"},'
        ' {"bbox_2d": [1, 2, 3], "label": "car"}]'
    )
    assert len(dets) == 1
    assert tuple(dets[0].box) == (0.0, 900.0, 2000.0, 1000.0)


def test_truncated_reply_keeps_complete_objects() -> None:
    dets = _parse('[{"bbox_2d": [0, 0, 500, 500], "label": "car"}, {"bbox_2d": [10, 20, 30')
    assert [d.label for d in dets] == ["car"]
