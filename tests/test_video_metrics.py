from datetime import UTC, datetime

import pytest

from discern.eval.types import GroundTruthBox
from discern.eval.video_metrics import (
    box_f1_at_frames,
    count_accuracy,
    count_mae,
    gpu_seconds_per_video_second,
    temporal_iou,
    verifier_pass_rate,
)
from discern.query.schemas import TimeRange, TrackFact
from discern.trace import TraceEvent
from discern.vision.boxes import Box


def _event(node: str, fallback: bool = False, gpu_ms: float = 0.0) -> TraceEvent:
    return TraceEvent(
        node=node,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        duration_ms=1.0,
        gpu_ms=gpu_ms,
        fallback_used=fallback,
    )


def test_count_accuracy_and_mae() -> None:
    predicted, expected = [3, 2, None, 5], [3, 4, 1, 5]
    assert count_accuracy(predicted, expected) == 0.5
    assert count_mae(predicted, expected) == pytest.approx((0 + 2 + 1 + 0) / 4)


def test_count_metrics_empty_and_mismatched() -> None:
    assert count_accuracy([], []) == 0.0 and count_mae([], []) == 0.0
    with pytest.raises(ValueError):
        count_accuracy([1], [1, 2])


def test_temporal_iou_partial_overlap() -> None:
    # [0, 4] vs [2, 6]: intersection 2, union 6
    assert temporal_iou([TimeRange(start=0, end=4)], [TimeRange(start=2, end=6)]) == pytest.approx(
        1 / 3
    )


def test_temporal_iou_merges_overlapping_ranges() -> None:
    pred = [TimeRange(start=0, end=3), TimeRange(start=2, end=4)]  # merges to [0, 4]
    assert temporal_iou(pred, [TimeRange(start=0, end=4)]) == 1.0


def test_temporal_iou_disjoint_and_empty() -> None:
    assert temporal_iou([TimeRange(start=0, end=1)], [TimeRange(start=2, end=3)]) == 0.0
    assert temporal_iou([], []) == 1.0
    assert temporal_iou([], [TimeRange(start=0, end=1)]) == 0.0
    # zero-length ranges that coincide have no duration to compare: treated as disagreement
    assert temporal_iou([TimeRange(start=1, end=1)], [TimeRange(start=1, end=1)]) == 0.0


def test_box_f1_at_sampled_frames() -> None:
    truth = {
        0: [GroundTruthBox(box=Box(0, 0, 10, 10), label="car")],
        5: [
            GroundTruthBox(box=Box(0, 0, 10, 10), label="car"),
            GroundTruthBox(box=Box(20, 20, 30, 30), label="car"),
        ],
    }
    tracks = [
        TrackFact(
            track_id=1,
            label="car",
            t_start=0,
            t_end=1,
            boxes={0: Box(0, 0, 10, 10), 5: Box(0, 0, 10, 10), 9: Box(50, 50, 60, 60)},
        ),
        # wrong label at frame 0: false positive there, and a missed box at frame 5 stays missed
        TrackFact(track_id=2, label="person", t_start=0, t_end=1, boxes={0: Box(0, 0, 10, 10)}),
    ]
    counts = box_f1_at_frames(tracks, truth)
    # frame 0: tp 1, fp 1 (person). frame 5: tp 1, fn 1. frame 9 is not sampled.
    assert (counts.tp, counts.fp, counts.fn) == (2, 1, 1)
    assert counts.f1 == pytest.approx(2 * (2 / 3) * (2 / 3) / (4 / 3))


def test_verifier_pass_rate() -> None:
    events = [
        _event("answer.verify"),
        _event("answer.verify", fallback=True),
        _event("answer.verify"),
        _event("answer.verify"),
        _event("detect", fallback=True),  # not a verification event
    ]
    assert verifier_pass_rate(events) == 0.75
    assert verifier_pass_rate([_event("detect")]) is None


def test_gpu_seconds_per_video_second() -> None:
    events = [_event("detect", gpu_ms=1500.0), _event("caption", gpu_ms=500.0)]
    assert gpu_seconds_per_video_second(events, 8.0) == 0.25
    with pytest.raises(ValueError):
        gpu_seconds_per_video_second(events, 0.0)
