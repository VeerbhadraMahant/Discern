"""Video QA metrics (system-design 10.2). Pure functions over plain values."""

from collections.abc import Mapping, Sequence

from discern.eval.metrics import Counts, match_image
from discern.eval.types import GroundTruthBox
from discern.models.roles import Detection
from discern.query.answer import VERIFY_NODE
from discern.query.schemas import TimeRange, TrackFact
from discern.trace import TraceEvent


def count_accuracy(predicted: Sequence[int | None], expected: Sequence[int]) -> float:
    """Share of questions whose predicted count equals the expected count exactly.
    A missing prediction (None) is wrong."""
    _same_length(predicted, expected)
    if not expected:
        return 0.0
    return sum(p == e for p, e in zip(predicted, expected, strict=True)) / len(expected)


def count_mae(predicted: Sequence[int | None], expected: Sequence[int]) -> float:
    """Mean absolute count error. A missing prediction counts as zero."""
    _same_length(predicted, expected)
    if not expected:
        return 0.0
    errors = [abs((p or 0) - e) for p, e in zip(predicted, expected, strict=True)]
    return sum(errors) / len(errors)


def _same_length(a: Sequence[object], b: Sequence[object]) -> None:
    if len(a) != len(b):
        raise ValueError(f"{len(a)} predictions for {len(b)} expected values")


def _merge(ranges: Sequence[TimeRange]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted((r.start, r.end) for r in ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def temporal_iou(predicted: Sequence[TimeRange], expected: Sequence[TimeRange]) -> float:
    """Intersection over union of the time covered by two sets of ranges (each set is merged
    first). Two empty sets agree perfectly; an empty set against a non-empty one scores 0."""
    a, b = _merge(predicted), _merge(expected)
    length_a, length_b = sum(e - s for s, e in a), sum(e - s for s, e in b)
    inter = sum(
        max(0.0, min(ae, be) - max(as_, bs)) for as_, ae in a for bs, be in b
    )
    union = length_a + length_b - inter
    if union <= 0:
        return 1.0 if not a and not b else 0.0
    return inter / union


def box_f1_at_frames(
    predicted: Sequence[TrackFact],
    expected: Mapping[int, Sequence[GroundTruthBox]],
    iou_threshold: float = 0.5,
) -> Counts:
    """Counts of F1@IoU over the sampled frames in `expected` (frame index -> ground-truth
    boxes). Predicted tracks contribute their box at each of those frames, labelled with the
    track label and a constant score, and matching is the image-level greedy matching."""
    total = Counts()
    for frame, truth in expected.items():
        preds = [
            Detection(box=t.boxes[frame], label=t.label, score=1.0, detector="video")
            for t in predicted
            if frame in t.boxes
        ]
        total += match_image(preds, truth, iou_threshold)
    return total


def verifier_pass_rate(events: Sequence[TraceEvent]) -> float | None:
    """Share of verified answers that passed without falling back to the template.
    None when the trace holds no verification event."""
    verified = [e for e in events if e.node == VERIFY_NODE]
    if not verified:
        return None
    return sum(not e.fallback_used for e in verified) / len(verified)


def gpu_seconds_per_video_second(events: Sequence[TraceEvent], video_seconds: float) -> float:
    """GPU time recorded in the trace divided by the duration of the video it processed."""
    if video_seconds <= 0:
        raise ValueError("video_seconds must be positive")
    return sum(e.gpu_ms for e in events) / 1000.0 / video_seconds
