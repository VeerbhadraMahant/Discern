from datetime import UTC, datetime

import pytest

from discern.config import load_settings
from discern.experience.aggregate import Memory, build_memory
from discern.experience.schema import ExperienceRecord
from discern.monitor.summary import SessionRecord, render_markdown, summarise
from discern.trace import TraceEvent

WEIGHTS = load_settings("local_lite").thresholds.experience.similarity_weights
MEMORY_KEY = "fog|dim|poor|small|dense"


def event(node: str, gpu_ms: float, fallback: bool) -> TraceEvent:
    return TraceEvent(
        node=node,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        duration_ms=1.0,
        gpu_ms=gpu_ms,
        fallback_used=fallback,
    )


def memory() -> Memory:
    record = ExperienceRecord(
        profile_key=MEMORY_KEY,
        query_type="count",
        node="restorer",
        option="none",
        metric_name="f1_best",
        metric_value=0.5,
        sample_id="s1",
        source="benchmark",
        memory_version="v1",
    )
    return build_memory([record], "v1")


RECORDS = [
    SessionRecord(
        query_type="count",
        profile_key="fog|dark|poor|small|dense",  # similarity 8/9 to the memory key: close
        events=[event("restorer", 1000, False), event("detector", 2000, True)],
    ),
    SessionRecord(
        query_type="count",
        profile_key="normal|bright|clear|large|sparse",  # other scene label: similarity 0
        events=[event("detector", 3000, False)],
    ),
    SessionRecord(query_type="temporal", grounded=False, events=[event("restorer", 500, True)]),
    SessionRecord(query_type="count", profile_key="fog|bright|clear|small|dense"),  # 4/9
]


def test_summary_matches_hand_computation() -> None:
    s = summarise(RECORDS, memory(), WEIGHTS, 0.8)
    assert s.requests == 4
    assert s.gpu_seconds_by_query_type == {"count": 6.0, "temporal": 0.5}
    assert s.fallback_rate_by_node == {"detector": 0.5, "restorer": 0.5}
    assert s.profile_distribution == {
        "fog|bright|clear|small|dense": 1,
        "fog|dark|poor|small|dense": 1,
        "normal|bright|clear|large|sparse": 1,
    }
    assert s.ungrounded_queries == 1
    assert (s.profiled_uploads, s.drifted_uploads) == (3, 2)
    assert s.drift_share == pytest.approx(2 / 3)


def test_threshold_moves_the_drift_share() -> None:
    assert summarise(RECORDS, memory(), WEIGHTS, 0.4).drifted_uploads == 1  # only scene mismatch
    assert summarise(RECORDS, memory(), WEIGHTS, 0.95).drifted_uploads == 3


def test_empty_inputs() -> None:
    s = summarise([], memory(), WEIGHTS, 0.8)
    assert s.requests == 0 and s.drift_share is None and s.gpu_seconds_by_query_type == {}


def test_render_markdown() -> None:
    text = render_markdown(summarise(RECORDS, memory(), WEIGHTS, 0.8))
    assert "| count | 6.00 |" in text
    assert "| detector | 50% |" in text
    assert "2 of 3 (67%)" in text
