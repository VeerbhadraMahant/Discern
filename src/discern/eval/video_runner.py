"""Run a video question set through a caller-supplied answer function and score it.

The function stands in for the whole system (a session over a real video on the GPU gate, a
fake in tests): `answer_fn(question) -> (answer_text, facts, trace_events)`. Questions of one
chain must be consecutive and in step order, as `flatten_chains` yields them, so a stateful
session sees the follow-ups in order.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean

from discern.config.settings import QueryThresholds
from discern.eval.metrics import Counts
from discern.eval.qa_templates import VideoQuestion, ground_truth_by_frame
from discern.eval.video_metrics import (
    box_f1_at_frames,
    count_accuracy,
    count_mae,
    gpu_seconds_per_video_second,
    temporal_iou,
    verifier_pass_rate,
)
from discern.query.schemas import Facts
from discern.trace import TraceEvent

AnswerFn = Callable[[VideoQuestion], tuple[str, Facts, Sequence[TraceEvent]]]

IOU_PASS = 0.5  # a relation answer is correct when its temporal IoU reaches this (as F1@0.5)


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    kind: str
    correct: bool
    answer: str
    gpu_seconds_per_video_second: float


@dataclass(frozen=True)
class VideoEvalSummary:
    """Aggregates over a question set. A metric is None when no question of its kind was asked."""

    n_questions: int
    count_accuracy: float | None = None
    count_mae: float | None = None
    presence_accuracy: float | None = None
    first_appearance_accuracy: float | None = None
    temporal_iou: float | None = None
    box_f1: float | None = None
    verifier_pass_rate: float | None = None
    chain_success_rate: float | None = None
    gpu_seconds_per_video_second: float | None = None
    results: tuple[QuestionResult, ...] = field(default=(), repr=False)

    def as_metrics(self) -> dict[str, float]:
        """Defined metrics only, named as the gate config names them."""
        names = (
            "count_accuracy",
            "count_mae",
            "presence_accuracy",
            "first_appearance_accuracy",
            "temporal_iou",
            "box_f1",
            "verifier_pass_rate",
            "chain_success_rate",
            "gpu_seconds_per_video_second",
        )
        values = {n: getattr(self, n) for n in names}
        return {n: float(v) for n, v in values.items() if v is not None}


def _first_appearance(facts: Facts) -> float | None:
    return min((t.t_start for t in facts.tracks), default=None)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def run_video_eval(
    questions: Sequence[VideoQuestion],
    answer_fn: AnswerFn,
    durations: Mapping[str, float],
    th: QueryThresholds,
) -> VideoEvalSummary:
    """Ask every question in order and score the facts each answer returns.

    `durations` maps video id to video seconds, for the GPU cost metric.
    """
    results: list[QuestionResult] = []
    count_pred: list[int | None] = []
    count_true: list[int] = []
    presence: list[bool] = []
    first_hits: list[bool] = []
    ious: list[float] = []
    box_counts = Counts()
    asked_box = False
    events: list[TraceEvent] = []
    for q in questions:
        text, facts, trace = answer_fn(q)
        events += trace
        exp = q.expected
        if q.kind == "count":
            assert exp.count is not None
            count_pred.append(facts.count)
            count_true.append(exp.count)
            correct = facts.count == exp.count
        elif q.kind == "presence":
            predicted = (facts.count if facts.count is not None else len(facts.tracks)) > 0
            correct = predicted == exp.present
            presence.append(correct)
        elif q.kind == "first_appearance":
            assert exp.time is not None
            found = _first_appearance(facts)
            correct = found is not None and abs(found - exp.time) <= th.time_tolerance_seconds
            first_hits.append(correct)
        elif q.kind == "relation":
            iou = temporal_iou(facts.time_ranges, exp.time_ranges)
            ious.append(iou)
            correct = iou >= IOU_PASS
        else:  # locate
            counts = box_f1_at_frames(facts.tracks, ground_truth_by_frame(exp.boxes))
            box_counts += counts
            asked_box = True
            correct = counts.f1 == 1.0
        results.append(
            QuestionResult(
                question_id=q.id,
                kind=q.kind,
                correct=correct,
                answer=text,
                gpu_seconds_per_video_second=gpu_seconds_per_video_second(
                    trace, durations[q.video_id]
                ),
            )
        )
    by_chain: dict[str, list[bool]] = {}
    for q, r in zip(questions, results, strict=True):
        if q.chain_id is not None:
            by_chain.setdefault(q.chain_id, []).append(r.correct)
    return VideoEvalSummary(
        n_questions=len(questions),
        count_accuracy=count_accuracy(count_pred, count_true) if count_true else None,
        count_mae=count_mae(count_pred, count_true) if count_true else None,
        presence_accuracy=_mean_or_none([float(c) for c in presence]),
        first_appearance_accuracy=_mean_or_none([float(c) for c in first_hits]),
        temporal_iou=_mean_or_none(ious),
        box_f1=box_counts.f1 if asked_box else None,
        verifier_pass_rate=verifier_pass_rate(events),
        chain_success_rate=_mean_or_none([float(all(v)) for v in by_chain.values()]),
        gpu_seconds_per_video_second=_mean_or_none(
            [r.gpu_seconds_per_video_second for r in results]
        ),
        results=tuple(results),
    )
