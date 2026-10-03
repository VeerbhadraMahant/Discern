from discern.models.fakes import FakeVLM
from discern.query.answer import UNGROUNDED_LABEL, verifier_counts
from discern.query.schemas import ConversationState
from discern.query.session import answer_question
from discern.trace import TraceCollector
from tests.query_fakes import SpyProvider, make_services, make_track, still

COUNT_PLAN = '{"query_type": "count", "targets": ["car"]}'
FILTER_PLAN = (
    '{"query_type": "refine", "source_result_set": "R1", "operations":'
    ' [{"kind": "filter_attribute", "attribute": "color", "value": "red"}]}'
)
LOCATE_PLAN = '{"query_type": "locate", "source_result_set": "R2"}'


def answer(text: str) -> str:
    return f'{{"answer": "{text}"}}'


def three_cars() -> SpyProvider:
    return SpyProvider(
        {
            "car": [
                make_track(1, "car", still((10, 10, 50, 50), 4), t0=0.0),
                make_track(2, "car", still((60, 10, 100, 50), 4), t0=1.0),
                make_track(3, "car", still((110, 10, 150, 50), 4), t0=2.0),
            ]
        }
    )


def test_follow_up_chain_reuses_tracks_without_re_detection() -> None:
    """count, then filter, then locate: detection runs exactly once."""
    provider = three_cars()
    vlm = FakeVLM(
        [
            COUNT_PLAN,
            answer("I counted 3 distinct car."),
            FILTER_PLAN,
            '{"value": "red"}',
            '{"value": "blue"}',
            '{"value": "red"}',
            answer("2 cars remain: track 1 and track 3."),
            LOCATE_PLAN,
            answer("Found 2 cars: track 1 from 0.0s to 1.5s and track 3 from 2.0s to 3.5s."),
        ]
    )
    trace, state = TraceCollector(), ConversationState()
    services = make_services(vlm, provider, trace)

    r1, text1 = answer_question(state, "How many cars?", services)
    r2, text2 = answer_question(state, "How many of them are red?", services)
    r3, text3 = answer_question(state, "Where are they?", services)

    assert r1 is not None and r2 is not None and r3 is not None
    assert provider.calls == [(("car",), False)]  # one scan for the whole chain
    assert r1.facts.count == 3 and r2.facts.count == 2
    assert r2.parent == "R1" and r3.parent == "R2"
    assert r3.track_ids == r2.track_ids == [1, 3]
    assert r3.facts.query_type == "locate" and r3.facts.tracks[0].t_end == 1.5
    assert text1 == "I counted 3 distinct car."
    assert "R3" in state.result_sets and len(state.turns) == 3
    assert state.aliases == {"red car": "R2"}
    assert verifier_counts(trace) == (3, 0)
    detection_nodes = [e.node for e in trace.events if e.node.startswith("execute.")]
    assert detection_nodes == ["execute.count", "execute.refine", "execute.locate"]


def test_a_repeated_full_scan_is_served_from_the_conversation_cache() -> None:
    provider = three_cars()
    vlm = FakeVLM([COUNT_PLAN, answer("3 car."), COUNT_PLAN, answer("3 car.")])
    state = ConversationState()
    services = make_services(vlm, provider)
    answer_question(state, "How many cars?", services)
    again, _ = answer_question(state, "And how many cars in total?", services)
    assert again is not None and again.facts.count == 3
    assert len(provider.calls) == 1


def test_track_ids_are_unique_across_the_conversation() -> None:
    provider = SpyProvider({"car": [make_track(1, "car", still((0, 0, 9, 9), 2))],
                            "person": [make_track(1, "person", still((0, 0, 9, 9), 2))]})
    vlm = FakeVLM(
        [COUNT_PLAN, answer("1 car."),
         '{"query_type": "count", "targets": ["person"]}', answer("1 person.")]
    )
    state = ConversationState()
    services = make_services(vlm, provider)
    r1, _ = answer_question(state, "cars?", services)
    r2, _ = answer_question(state, "people?", services)
    assert r1 is not None and r2 is not None
    assert r1.track_ids == [1] and r2.track_ids == [2]


def test_unresolvable_reference_asks_a_clarifying_question_and_runs_nothing() -> None:
    provider = three_cars()
    vlm = FakeVLM([FILTER_PLAN])  # there is no R1 yet; no second VLM call is scripted
    state = ConversationState()
    result, text = answer_question(state, "how many of them are red?", make_services(vlm, provider))
    assert result is None
    assert text.startswith("Which earlier result do you mean?")
    assert provider.calls == [] and state.result_sets == {}
    assert state.turns[0].result_set_id is None and state.turns[0].answer == text


def test_unknown_result_id_with_existing_results_lists_them() -> None:
    provider = three_cars()
    vlm = FakeVLM([COUNT_PLAN, answer("3 car."), FILTER_PLAN.replace("R1", "R5")])
    state = ConversationState()
    services = make_services(vlm, provider)
    answer_question(state, "How many cars?", services)
    result, text = answer_question(state, "how many are red?", services)
    assert result is None and "R1" in text and len(state.result_sets) == 1


def test_invalid_json_everywhere_degrades_to_a_labelled_describe_answer() -> None:
    state = ConversationState()
    provider = SpyProvider({})
    vlm = FakeVLM(["bad", "bad", "bad", "bad"])  # parse and answer, each with its repair retry
    trace = TraceCollector()
    result, text = answer_question(state, "what is going on?", make_services(vlm, provider, trace))
    assert result is not None and result.plan.query_type == "describe"
    assert result.plan.question == "what is going on?"
    assert text.startswith(UNGROUNDED_LABEL) and "a road" in text and not result.grounded
    assert provider.calls == []
    assert all(e.fallback_used for e in trace.events if e.node in ("query_parse", "answer"))


def test_wrong_number_in_a_session_answer_is_replaced_by_the_template() -> None:
    trace = TraceCollector()
    vlm = FakeVLM([COUNT_PLAN, answer("I counted 9 cars.")])
    result, text = answer_question(
        ConversationState(), "How many cars?", make_services(vlm, three_cars(), trace)
    )
    assert result is not None
    assert text == (
        "I counted 3 distinct car. track 1 (car) 0.0s to 1.5s; track 2 (car) 1.0s to 2.5s; "
        "track 3 (car) 2.0s to 3.5s."
    )
    assert verifier_counts(trace) == (0, 1)


def test_locate_question_uses_retrieval_then_answers_from_facts() -> None:
    provider = three_cars()
    vlm = FakeVLM(
        ['{"query_type": "locate", "targets": ["car"]}', answer("Found 3 object(s) matching car.")]
    )
    result, text = answer_question(
        ConversationState(), "Where are the cars?", make_services(vlm, provider)
    )
    assert provider.calls == [(("car",), True)]
    assert result is not None and result.facts.count == 3 and "3" in text
