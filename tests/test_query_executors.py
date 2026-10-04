from pathlib import Path

import pytest

from discern.agent.llm_io import load_prompt
from discern.config.settings import load_settings
from discern.models.fakes import FakeVLM
from discern.query.executors import Execution, _track_fact, execute, relation_holds
from discern.query.schemas import (
    ConversationState,
    Facts,
    Operation,
    Predicate,
    QueryPlan,
    Relation,
    ResultSet,
    TimeRange,
)
from discern.trace import TraceCollector
from discern.video.types import Track
from discern.vision.boxes import Box
from tests.query_fakes import SpyProvider, make_index, make_services, make_track, still

SNAPSHOTS = Path(__file__).parent / "snapshots"
TH = load_settings("local_lite").thresholds.query
CAR = (100.0, 40.0, 140.0, 80.0)


def cars() -> list[Track]:  # three accepted cars and one rejected
    return [
        make_track(1, "car", still(CAR, 4), t0=0.0),
        make_track(2, "car", still((10, 10, 50, 50), 6), t0=2.0),
        make_track(3, "car", still((150, 60, 190, 90), 2), t0=6.0),
        make_track(4, "car", still(CAR, 3), t0=1.0, status="rejected"),
    ]


def state_with(tracks: list[Track]) -> ConversationState:
    state = ConversationState()
    state.result_sets["R1"] = ResultSet(
        id="R1",
        question="how many cars",
        plan=QueryPlan(query_type="count", targets=["car"]),
        tracks=tracks,
        facts=Facts(
            query_type="count",
            targets=["car"],
            count=len(tracks),
            tracks=[_track_fact(t) for t in tracks],
        ),
        answer="",
    )
    return state


def run(
    plan: QueryPlan,
    provider: SpyProvider,
    vlm: FakeVLM | None = None,
    state: ConversationState | None = None,
    trace: TraceCollector | None = None,
) -> Execution:
    services = make_services(vlm or FakeVLM([]), provider, trace)
    return execute(plan, services, state or ConversationState())


def test_attribute_prompt_matches_golden_snapshot() -> None:
    rendered = load_prompt("attribute_check", 1).render(label="car", attribute="color")
    assert rendered == (SNAPSHOTS / "attribute_check.v1.txt").read_text(encoding="utf-8")


# ---- locate -----------------------------------------------------------------------------------


def test_locate_reports_tracks_first_last_seen_and_boxes_per_frame() -> None:
    provider = SpyProvider({"car": cars()})
    out = run(QueryPlan(query_type="locate", targets=["car"]), provider)
    assert provider.calls == [(("car",), True)]  # retrieval first
    assert out.facts.grounded and out.facts.count == 3
    first = out.facts.tracks[0]
    assert (first.track_id, first.t_start, first.t_end) == (1, 0.0, 1.5)
    assert first.boxes[2] == Box(*CAR)
    assert len(first.boxes) == 4


def test_retrieval_is_traced_with_its_segments() -> None:
    trace = TraceCollector()
    run(QueryPlan(query_type="locate", targets=["car"]), SpyProvider({"car": cars()}), trace=trace)
    (event,) = [e for e in trace.events if e.node == "retrieve"]
    assert event.decision != "none" and "s-" in event.decision


def test_locate_falls_back_to_a_full_scan_when_segments_yield_nothing() -> None:
    provider = SpyProvider({"car": cars()}, empty_on_segments=True)
    trace = TraceCollector()
    out = run(QueryPlan(query_type="locate", targets=["car"]), provider, trace=trace)
    assert provider.calls == [(("car",), True), (("car",), False)]
    assert out.facts.count == 3
    assert any(e.node == "scan.fallback" for e in trace.events)


# ---- count ------------------------------------------------------------------------------------


def test_count_scans_everything_and_counts_distinct_accepted_tracks() -> None:
    provider = SpyProvider({"car": cars()})
    out = run(QueryPlan(query_type="count", targets=["car"]), provider)
    assert provider.calls == [(("car",), False)]  # never pruned by retrieval
    assert out.facts.count == 3 and [t.id for t in out.tracks] == [1, 2, 3]  # rejected excluded


def test_count_within_a_time_range_keeps_overlapping_tracks() -> None:
    plan = QueryPlan(
        query_type="count", targets=["car"], time_range=TimeRange(start=0.0, end=1.9)
    )
    assert run(plan, SpyProvider({"car": cars()})).facts.count == 1  # only track 1 starts early


def test_count_with_an_attribute_constraint_uses_the_attribute_check() -> None:
    plan = QueryPlan.model_validate(
        {
            "query_type": "count",
            "targets": ["car"],
            "attributes": [{"target": "car", "attribute": "color", "value": "red"}],
        }
    )
    vlm = FakeVLM(['{"value": "Red"}', '{"value": "blue"}', '{"value": "red"}'])
    out = run(plan, SpyProvider({"car": cars()}), vlm)
    assert [t.id for t in out.tracks] == [1, 3]


# ---- temporal ---------------------------------------------------------------------------------


def test_temporal_gives_the_time_range_of_each_track() -> None:
    out = run(QueryPlan(query_type="temporal", targets=["car"]), SpyProvider({"car": cars()}))
    assert out.facts.time_ranges[:2] == [
        TimeRange(start=0.0, end=1.5),
        TimeRange(start=2.0, end=4.5),
    ]


# ---- relation ---------------------------------------------------------------------------------


def person_then_car_scene() -> SpyProvider:
    person = [(10.0, 40.0, 30.0, 80.0)] * 4 + [(150.0, 40.0, 170.0, 80.0)] * 2
    return SpyProvider(
        {"person": [make_track(1, "person", person)], "car": [make_track(2, "car", still(CAR, 6))]}
    )


def relation_plan(predicate: Predicate) -> QueryPlan:
    return QueryPlan(
        query_type="relation",
        relations=[Relation(subject="person", predicate=predicate, object="car")],
    )


def test_relation_gives_pairs_and_the_time_ranges_where_it_holds() -> None:
    provider = person_then_car_scene()
    out = run(relation_plan("left_of"), provider)
    (pair,) = out.facts.pairs
    assert (pair.subject_track, pair.object_track) == (1, 2)
    assert pair.time_ranges == [TimeRange(start=0.0, end=1.5)]
    assert out.facts.count == 1 and {t.track_id for t in out.facts.tracks} == {1, 2}
    assert provider.calls == [(("person",), False), (("car",), False)]


def test_relation_near_holds_only_while_the_person_is_close() -> None:
    out = run(relation_plan("near"), person_then_car_scene())
    assert out.facts.pairs[0].time_ranges == [TimeRange(start=2.0, end=2.5)]


def test_relation_with_no_match_has_no_pairs() -> None:
    out = run(relation_plan("above"), person_then_car_scene())
    assert out.facts.pairs == [] and out.facts.count == 0


def test_a_track_is_never_paired_with_itself() -> None:
    plan = QueryPlan(
        query_type="relation",
        relations=[Relation(subject="car", predicate="near", object="car")],
    )
    out = run(plan, SpyProvider({"car": cars()[:2]}))
    assert all(p.subject_track != p.object_track for p in out.facts.pairs)


@pytest.mark.parametrize(
    ("predicate", "a", "b", "expected"),
    [
        ("left_of", Box(0, 0, 10, 10), Box(30, 0, 40, 10), True),
        ("left_of", Box(30, 0, 40, 10), Box(0, 0, 10, 10), False),
        ("right_of", Box(30, 0, 40, 10), Box(0, 0, 10, 10), True),
        ("above", Box(0, 0, 10, 10), Box(0, 30, 10, 40), True),
        ("below", Box(0, 30, 10, 40), Box(0, 0, 10, 10), True),
        ("below", Box(0, 0, 10, 10), Box(0, 30, 10, 40), False),
        ("overlapping", Box(0, 0, 10, 10), Box(5, 5, 15, 15), True),
        ("overlapping", Box(0, 0, 10, 10), Box(20, 20, 30, 30), False),
        ("near", Box(0, 0, 10, 10), Box(12, 0, 22, 10), True),
        ("near", Box(0, 0, 10, 10), Box(100, 0, 110, 10), False),
    ],
)
def test_geometric_predicates(predicate: Predicate, a: Box, b: Box, expected: bool) -> None:
    assert relation_holds(predicate, a, b, TH) is expected


# ---- describe ---------------------------------------------------------------------------------


def test_describe_is_ungrounded_has_empty_facts_and_runs_no_detection() -> None:
    provider = SpyProvider({})
    index = make_index(captions={0: "a road", 1: ""})
    services = make_services(FakeVLM([]), provider, index=index)
    plan = QueryPlan(query_type="describe", question="what is here")
    out = execute(plan, services, ConversationState())
    assert not out.facts.grounded and out.facts.count is None and out.facts.tracks == []
    assert out.context == "- a road"  # empty captions are skipped, not invented
    assert provider.calls == []


def test_describe_respects_a_time_range() -> None:
    plan = QueryPlan(query_type="describe", time_range=TimeRange(start=12.0, end=15.0))
    assert run(plan, SpyProvider({})).context == "- a parking lot"


# ---- refine -----------------------------------------------------------------------------------


def refine(*ops: Operation) -> QueryPlan:
    return QueryPlan(query_type="refine", source_result_set="R1", operations=list(ops))


def test_refine_filters_by_attribute_without_detection_and_caches_per_track() -> None:
    state = state_with(cars()[:3])
    provider = SpyProvider({})
    vlm = FakeVLM(['{"value": "red"}', '{"value": "blue"}', '{"value": "red"}'])
    op = Operation(kind="filter_attribute", attribute="color", value="red")
    out = run(refine(op), provider, vlm, state)
    assert [t.id for t in out.tracks] == [1, 3] and out.facts.count == 2
    assert provider.calls == []
    assert state.attribute_cache == {(1, "color"): "red", (2, "color"): "blue", (3, "color"): "red"}
    # same attribute again: answered from the cache (the empty FakeVLM would raise otherwise)
    blue = Operation(kind="filter_attribute", attribute="color", value="blue")
    again = run(refine(blue), provider, vlm, state)
    assert [t.id for t in again.tracks] == [2]
    assert len(vlm.prompts) == 3


def test_unclassifiable_tracks_are_excluded_and_not_cached() -> None:
    state = state_with(cars()[:1])
    op = Operation(kind="filter_attribute", attribute="color", value="red")
    out = run(refine(op), SpyProvider({}), FakeVLM(["bad", "bad"]), state)
    assert out.tracks == [] and state.attribute_cache == {}


def test_refine_filters_by_time() -> None:
    state = state_with(cars()[:3])
    op = Operation(kind="filter_time", time_range=TimeRange(start=5.0, end=9.0))
    assert [t.id for t in run(refine(op), SpyProvider({}), state=state).tracks] == [3]


def test_refine_filters_by_region() -> None:
    state = state_with(cars()[:3])  # frame is 200 wide: track 2 is left, tracks 1 and 3 right
    left = run(refine(Operation(kind="filter_region", region="left")), SpyProvider({}), state=state)
    assert [t.id for t in left.tracks] == [2]
    right = run(
        refine(Operation(kind="filter_region", region="right")), SpyProvider({}), state=state
    )
    assert [t.id for t in right.tracks] == [1, 3]


def test_refine_count_and_show_keep_the_set() -> None:
    state = state_with(cars()[:3])
    for op in (Operation(kind="count"), Operation(kind="show")):
        out = run(refine(op), SpyProvider({}), state=state)
        assert out.facts.count == 3 and len(out.facts.tracks) == 3


def test_refine_compare_reports_both_counts() -> None:
    state = state_with(cars()[:3])
    state.result_sets["R2"] = state.result_sets["R1"].model_copy(
        update={"id": "R2", "tracks": cars()[:1]}
    )
    out = run(refine(Operation(kind="compare", value="R2")), SpyProvider({}), state=state)
    assert out.facts.compared == {"R1": 3, "R2": 1}


def test_operations_apply_in_order() -> None:
    state = state_with(cars()[:3])
    ops = [
        Operation(kind="filter_region", region="right"),
        Operation(kind="filter_time", time_range=TimeRange(start=5.0, end=9.0)),
    ]
    assert [t.id for t in run(refine(*ops), SpyProvider({}), state=state).tracks] == [3]


def test_symmetric_relation_between_the_same_kind_is_one_pair_not_two() -> None:
    plan = QueryPlan(
        query_type="relation",
        relations=[Relation(subject="car", predicate="near", object="car")],
    )
    close = [
        make_track(1, "car", still(CAR, 4)),
        make_track(2, "car", still((110, 40, 150, 80), 4)),
    ]
    out = run(plan, SpyProvider({"car": close}))
    assert [(p.subject_track, p.object_track) for p in out.facts.pairs] == [(1, 2)]
    assert out.facts.count == 1
    # a directional predicate keeps both orders: 1 left_of 2 is not 2 left_of 1
    left = QueryPlan(
        query_type="relation",
        relations=[Relation(subject="car", predicate="left_of", object="car")],
    )
    far = [
        make_track(1, "car", still(CAR, 4)),
        make_track(2, "car", still((150, 40, 190, 80), 4)),
    ]
    assert run(left, SpyProvider({"car": far})).facts.count == 1


def test_relation_applies_attribute_constraints_to_the_entity_they_describe() -> None:
    plan = QueryPlan.model_validate(
        {
            "query_type": "relation",
            "relations": [{"subject": "person", "predicate": "left_of", "object": "car"}],
            "attributes": [{"target": "car", "attribute": "color", "value": "red"}],
        }
    )
    cars_ = [
        make_track(2, "car", still(CAR, 6)),
        make_track(3, "car", still((120, 40, 160, 80), 6)),
    ]
    person = make_track(1, "person", still((10.0, 40.0, 30.0, 80.0), 6))
    out = run(plan, SpyProvider({"person": [person], "car": cars_}),
              FakeVLM(['{"value": "blue"}', '{"value": "red"}']))
    assert [(p.subject_track, p.object_track) for p in out.facts.pairs] == [(1, 3)]
