import pytest

from discern.trace import TraceCollector


def test_span_records_event() -> None:
    trace = TraceCollector()
    with trace.span("perception") as s:
        s.decision = "fog"
        s.rationale = "low contrast"
        s.prompt_version = "perception@1"
    (event,) = trace.events
    assert event.node == "perception"
    assert event.decision == "fog"
    assert event.prompt_version == "perception@1"
    assert event.duration_ms >= 0
    assert event.fallback_used is False


def test_span_records_event_even_when_node_raises() -> None:
    trace = TraceCollector()
    with pytest.raises(RuntimeError), trace.span("detect"):
        raise RuntimeError("boom")
    assert [e.node for e in trace.events] == ["detect"]


def test_events_keep_order() -> None:
    trace = TraceCollector()
    for name in ("a", "b", "c"):
        with trace.span(name):
            pass
    assert [e.node for e in trace.events] == ["a", "b", "c"]
