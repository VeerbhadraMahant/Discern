from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from discern.config import load_settings
from discern.eval.qa_templates import (
    AnnotatedTrack,
    VideoAnnotation,
    VideoQuestion,
    flatten_chains,
    generate_chains,
    generate_questions,
)
from discern.eval.video_runner import run_video_eval
from discern.query.schemas import Facts, TimeRange, TrackFact
from discern.trace import TraceEvent
from discern.vision.boxes import Box

TH = load_settings("local_lite").thresholds.query


def _track(tid: int, label: str, frames: dict[int, float], x: float) -> AnnotatedTrack:
    return AnnotatedTrack(
        track_id=tid,
        label=label,
        boxes={f: Box(x, 0, x + 10, 10) for f in frames},
        timestamps=frames,
    )


ANN = VideoAnnotation(
    video_id="v1",
    duration=10.0,
    width=100,
    height=50,
    tracks=(
        _track(1, "car", {0: 0.0, 5: 1.0, 10: 2.0}, x=0),
        _track(2, "car", {50: 6.0, 60: 7.0}, x=80),
        _track(3, "person", {0: 0.0, 5: 1.0}, x=40),
    ),
)
QUESTIONS = generate_questions(ANN, TH, absent_labels=["train"]) + flatten_chains(
    generate_chains(ANN, TH)
)
DURATIONS = {"v1": 10.0}


def _events(fallback: bool = False) -> list[TraceEvent]:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        TraceEvent(node="detect", started_at=now, duration_ms=1.0, gpu_ms=2000.0),
        TraceEvent(node="answer.verify", started_at=now, duration_ms=1.0, fallback_used=fallback),
    ]


def oracle(q: VideoQuestion) -> tuple[str, Facts, Sequence[TraceEvent]]:
    """Answers every question correctly from its expected answer."""
    e = q.expected
    if q.kind == "count":
        facts = Facts(query_type="count", count=e.count)
    elif q.kind == "presence":
        facts = Facts(query_type="count", count=int(bool(e.present)))
    elif q.kind == "first_appearance":
        assert e.time is not None
        track = TrackFact(track_id=1, label="x", t_start=e.time, t_end=e.time + 1, boxes={})
        facts = Facts(query_type="temporal", tracks=[track])
    elif q.kind == "relation":
        facts = Facts(query_type="relation", time_ranges=list(e.time_ranges))
    else:
        tracks: dict[tuple[int, str], dict[int, Box]] = {}
        seen: dict[tuple[int, str], int] = {}
        for fb in e.boxes:  # the n-th box of a label in a frame belongs to its n-th track
            n = seen.get((fb.frame, fb.label), 0)
            seen[(fb.frame, fb.label)] = n + 1
            tracks.setdefault((n, fb.label), {})[fb.frame] = fb.box
        facts = Facts(
            query_type="locate",
            tracks=[
                TrackFact(track_id=i, label=lab, t_start=0, t_end=1, boxes=boxes)
                for (i, lab), boxes in tracks.items()
            ],
        )
    return "answer", facts, _events()


def test_perfect_system_scores_perfectly() -> None:
    s = run_video_eval(QUESTIONS, oracle, DURATIONS, TH)
    assert s.n_questions == len(QUESTIONS)
    assert s.count_accuracy == 1.0 and s.count_mae == 0.0
    assert s.presence_accuracy == 1.0 and s.first_appearance_accuracy == 1.0
    assert s.temporal_iou == 1.0 and s.box_f1 == 1.0
    assert s.verifier_pass_rate == 1.0
    assert s.chain_success_rate == 1.0
    assert s.gpu_seconds_per_video_second == pytest.approx(0.2)  # 2 s of GPU over 10 s of video
    assert all(r.correct for r in s.results)


def test_off_by_one_counts_lower_accuracy_and_break_chains() -> None:
    def off_by_one(q: VideoQuestion) -> tuple[str, Facts, Sequence[TraceEvent]]:
        text, facts, trace = oracle(q)
        if q.kind == "count" and facts.count is not None:
            facts = facts.model_copy(update={"count": facts.count + 1})
        return text, facts, trace

    s = run_video_eval(QUESTIONS, off_by_one, DURATIONS, TH)
    assert s.count_accuracy == 0.0
    assert s.count_mae == 1.0
    assert s.chain_success_rate == 0.0
    assert s.presence_accuracy == 1.0  # unaffected kinds still score


def test_verifier_fallbacks_and_missing_answers() -> None:
    def fallback_and_blank(q: VideoQuestion) -> tuple[str, Facts, Sequence[TraceEvent]]:
        _, facts, _ = oracle(q)
        if q.kind == "count":
            facts = Facts(query_type="count", count=None)
        return "", facts, _events(fallback=True)

    s = run_video_eval(QUESTIONS, fallback_and_blank, DURATIONS, TH)
    assert s.verifier_pass_rate == 0.0
    assert s.count_accuracy == 0.0


def test_wrong_relation_ranges_and_boxes_are_incorrect() -> None:
    def wrong(q: VideoQuestion) -> tuple[str, Facts, Sequence[TraceEvent]]:
        text, facts, trace = oracle(q)
        if q.kind == "relation":
            facts = Facts(query_type="relation", time_ranges=[TimeRange(start=8, end=9)])
        if q.kind == "locate":
            facts = Facts(query_type="locate")
        return text, facts, trace

    s = run_video_eval(QUESTIONS, wrong, DURATIONS, TH)
    assert s.temporal_iou == 0.0
    assert s.box_f1 == 0.0


def test_metrics_dict_has_only_defined_metrics() -> None:
    counts_only = [q for q in QUESTIONS if q.kind == "count" and q.chain_id is None]
    s = run_video_eval(counts_only, oracle, DURATIONS, TH)
    m = s.as_metrics()
    assert m["count_accuracy"] == 1.0
    assert "box_f1" not in m and "temporal_iou" not in m and "chain_success_rate" not in m
