from pathlib import Path

import pytest

from discern.agent.llm_io import load_prompt
from discern.agent.nodes.query_parse import Clarification, query_parse
from discern.models.fakes import FakeVLM
from discern.query.schemas import QueryPlan, ResultSetSummary, TimeRange
from discern.trace import TraceCollector

SNAPSHOTS = Path(__file__).parent / "snapshots"
R1 = ResultSetSummary(
    id="R1",
    question="how many cars",
    targets=["car"],
    count=3,
    time_range=TimeRange(start=0.0, end=9.5),
    aliases=[],
)
RESULTS_LINE = (
    'R1: question="how many cars", targets=car, count=3, time span=0.0s to 9.5s, aliases=none'
)
REFINE = (
    '{"query_type": "refine", "source_result_set": "R1", "operations":'
    ' [{"kind": "filter_attribute", "attribute": "color", "value": "red"}]}'
)


def test_prompt_matches_golden_snapshot() -> None:
    rendered = load_prompt("query_parse", 1).render(
        question="how many of them are red?", results=RESULTS_LINE
    )
    assert rendered == (SNAPSHOTS / "query_parse.v1.txt").read_text(encoding="utf-8")


def test_new_question_becomes_a_plan_carrying_the_raw_question() -> None:
    vlm = FakeVLM(['{"query_type": "count", "targets": ["car"]}'])
    plan = query_parse(vlm, TraceCollector(), "How many cars?", [])
    assert isinstance(plan, QueryPlan)
    assert plan.query_type == "count" and plan.targets == ["car"]
    assert plan.question == "How many cars?"
    assert "none" in vlm.prompts[0]  # no earlier results


def test_follow_up_resolves_to_an_existing_result_set_and_sees_its_summary() -> None:
    vlm = FakeVLM([REFINE])
    plan = query_parse(vlm, TraceCollector(), "how many are red?", [R1])
    assert isinstance(plan, QueryPlan) and plan.source_result_set == "R1"
    assert plan.operations[0].kind == "filter_attribute"
    assert RESULTS_LINE in vlm.prompts[0]


def test_unknown_result_set_asks_for_clarification() -> None:
    trace = TraceCollector()
    vlm = FakeVLM([REFINE.replace("R1", "R9")])
    result = query_parse(vlm, trace, "how many are red?", [R1])
    assert isinstance(result, Clarification)
    assert "R1" in result.message  # lists what is available
    guard = [e for e in trace.events if e.node == "query_parse.guard"]
    assert guard and guard[0].decision == "clarification"


def test_refine_without_a_source_asks_for_clarification() -> None:
    vlm = FakeVLM(['{"query_type": "refine", "operations": [{"kind": "show"}]}'])
    result = query_parse(vlm, TraceCollector(), "show them", [])
    assert isinstance(result, Clarification)
    assert "ask a new question first" in result.message


def test_compare_with_a_missing_result_set_asks_for_clarification() -> None:
    reply = (
        '{"query_type": "refine", "source_result_set": "R1",'
        ' "operations": [{"kind": "compare", "value": "R7"}]}'
    )
    result = query_parse(FakeVLM([reply]), TraceCollector(), "compare", [R1])
    assert isinstance(result, Clarification)


def test_invalid_output_falls_back_to_a_describe_query() -> None:
    trace = TraceCollector()
    plan = query_parse(FakeVLM(["not json", "still not json"]), trace, "what is going on?", [])
    assert plan == QueryPlan(query_type="describe", question="what is going on?")
    assert trace.events[0].fallback_used


def test_unknown_query_type_gets_one_repair_retry() -> None:
    vlm = FakeVLM(
        [
            '{"query_type": "banana", "targets": ["car"]}',
            '{"query_type": "locate", "targets": ["car"]}',
        ]
    )
    plan = query_parse(vlm, TraceCollector(), "where are cars", [])
    assert isinstance(plan, QueryPlan) and plan.query_type == "locate"
    assert len(vlm.prompts) == 2 and "Validation error" in vlm.prompts[1]


@pytest.mark.parametrize(
    "reply",
    [
        '{"query_type": "refine", "source_result_set": "R1", "operations": []}',
        '{"query_type": "count"}',
        '{"query_type": "relation"}',
        '{"query_type": "refine", "source_result_set": "R1", "operations":'
        ' [{"kind": "filter_region"}]}',
    ],
)
def test_structurally_incomplete_plans_are_rejected_by_validation(reply: str) -> None:
    plan = query_parse(FakeVLM([reply, reply]), TraceCollector(), "q", [R1])
    assert isinstance(plan, QueryPlan) and plan.query_type == "describe"
