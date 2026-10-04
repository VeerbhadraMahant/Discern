from pathlib import Path

import pytest

from discern.agent.llm_io import load_prompt
from discern.config.settings import load_settings
from discern.models.fakes import FakeVLM
from discern.query.answer import (
    UNGROUNDED_LABEL,
    compose_answer,
    extract_numbers,
    facts_summary,
    template_answer,
    verifier_counts,
    verify,
)
from discern.query.schemas import Facts, PairFact, TimeRange, TrackFact
from discern.trace import TraceCollector
from discern.vision.boxes import Box

SNAPSHOTS = Path(__file__).parent / "snapshots"
TOLERANCE = load_settings("local_lite").thresholds.query.time_tolerance_seconds
BOX = Box(0, 0, 10, 10)


def track_fact(track_id: int, t_start: float, t_end: float) -> TrackFact:
    return TrackFact(
        track_id=track_id, label="car", t_start=t_start, t_end=t_end, boxes={0: BOX}
    )


COUNT_FACTS = Facts(
    query_type="count",
    targets=["car"],
    count=2,
    tracks=[track_fact(1, 0.0, 3.2), track_fact(2, 1.0, 4.5)],
)
LOCATE_FACTS = COUNT_FACTS.model_copy(update={"query_type": "locate"})
RELATION_FACTS = Facts(
    query_type="relation",
    targets=["person", "car"],
    count=1,
    relation="person near car",
    tracks=[track_fact(1, 0.0, 3.2), track_fact(2, 0.0, 4.0)],
    time_ranges=[TimeRange(start=1.0, end=2.5)],
    pairs=[PairFact(subject_track=1, object_track=2, time_ranges=[TimeRange(start=1.0, end=2.5)])],
)
DESCRIBE_FACTS = Facts(query_type="describe", grounded=False)


def answer(text: str) -> str:
    return f'{{"answer": "{text}"}}'


def compose(vlm: FakeVLM, facts: Facts, trace: TraceCollector, context: str = "") -> str:
    return compose_answer(vlm, trace, "How many cars?", facts, context)


def test_prompt_matches_golden_snapshot() -> None:
    rendered = load_prompt("answer", 1).render(
        question="How many cars are there?", facts=facts_summary(COUNT_FACTS), context="none"
    )
    assert rendered == (SNAPSHOTS / "answer.v1.txt").read_text(encoding="utf-8")


# ---- number and timestamp extraction ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "times", "numbers"),
    [
        ("seen from 3.2s to 4.5s", [3.2, 4.5], []),
        ("at 1:05 and 0:07.5", [65.0, 7.5], []),
        ("for 4 seconds, then 2 minutes", [4.0, 120.0], []),
        ("two cars and track 3", [], [2.0, 3.0]),
        ("I counted 12 cars", [], [12.0]),
        ("result R3 on the 1st track", [], []),
        ("no numbers here", [], []),
    ],
)
def test_extract_numbers(text: str, times: list[float], numbers: list[float]) -> None:
    got_times, got_numbers = extract_numbers(text)
    assert got_times == pytest.approx(times)
    assert sorted(got_numbers) == sorted(numbers)


# ---- verifier ---------------------------------------------------------------------------------


def test_verify_accepts_numbers_and_times_from_the_facts() -> None:
    text = "There are 2 cars: track 1 from 0.0s to 3.2s and track 2 from 1.0s to 4.5s."
    assert verify(text, COUNT_FACTS, TOLERANCE) == []


def test_verify_catches_a_wrong_count() -> None:
    assert verify("There are 3 cars.", COUNT_FACTS, TOLERANCE) == ["number 3 is not in the facts"]
    assert verify("There are three cars.", COUNT_FACTS, TOLERANCE)


def test_verify_catches_a_wrong_timestamp() -> None:
    problems = verify("Track 1 is seen until 9.9s.", COUNT_FACTS, TOLERANCE)
    assert problems == ["time 9.9s is not in the facts"]


def test_verify_allows_times_within_the_configured_tolerance_only() -> None:
    assert verify(f"until {1.0 + TOLERANCE - 0.1:.1f}s", COUNT_FACTS, TOLERANCE) == []
    assert verify("until 2.2s", COUNT_FACTS, TOLERANCE)  # 1.0s from the nearest computed time


def test_verify_allows_durations_and_relation_ranges() -> None:
    assert verify("for 1.5 seconds", RELATION_FACTS, TOLERANCE) == []  # 2.5s minus 1.0s
    assert verify("track 1 near track 2 from 1.0s to 2.5s", RELATION_FACTS, TOLERANCE) == []


def test_ungrounded_text_may_contain_no_numbers() -> None:
    assert verify("A road with cars.", DESCRIBE_FACTS, TOLERANCE) == []
    assert verify("A road with 4 cars.", DESCRIBE_FACTS, TOLERANCE)


@pytest.mark.parametrize("facts", [COUNT_FACTS, LOCATE_FACTS, RELATION_FACTS])
def test_template_answers_always_pass_the_verifier(facts: Facts) -> None:
    assert verify(template_answer(facts), facts, TOLERANCE) == []


def test_empty_results_have_a_template_that_passes() -> None:
    for kind in ("locate", "count", "temporal", "refine", "relation"):
        facts = Facts(query_type=kind, targets=["car"], count=0)  # type: ignore[arg-type]
        text = template_answer(facts)
        assert verify(text, facts, TOLERANCE) == [], text


# ---- compose_answer ---------------------------------------------------------------------------


def test_verified_answer_is_used_as_written() -> None:
    trace = TraceCollector()
    text = "There are 2 cars, seen from 0.0s to 4.5s."
    assert compose(FakeVLM([answer(text)]), COUNT_FACTS, trace) == text
    assert verifier_counts(trace) == (1, 0)


def test_wrong_number_falls_back_to_the_template() -> None:
    trace = TraceCollector()
    out = compose(FakeVLM([answer("There are 5 cars.")]), COUNT_FACTS, trace)
    assert out == template_answer(COUNT_FACTS)
    assert verifier_counts(trace) == (0, 1)
    event = [e for e in trace.events if e.node == "answer.verify"][0]
    assert event.fallback_used and "5" in event.rationale


def test_wrong_timestamp_falls_back_to_the_template() -> None:
    trace = TraceCollector()
    out = compose(FakeVLM([answer("2 cars; the first leaves at 12.0s.")]), COUNT_FACTS, trace)
    assert out == template_answer(COUNT_FACTS)
    assert verifier_counts(trace) == (0, 1)


def test_invalid_json_uses_the_template() -> None:
    trace = TraceCollector()
    out = compose(FakeVLM(["oops", "oops again"]), COUNT_FACTS, trace)
    assert out == template_answer(COUNT_FACTS)
    assert trace.events[0].fallback_used  # the structured call itself fell back


def test_counts_accumulate_over_several_answers() -> None:
    trace = TraceCollector()
    vlm = FakeVLM([answer("2 cars."), answer("7 cars."), answer("2 cars.")])
    for _ in range(3):
        compose(vlm, COUNT_FACTS, trace)
    assert verifier_counts(trace) == (2, 1)


def test_describe_answers_are_labelled_ungrounded() -> None:
    trace = TraceCollector()
    out = compose(FakeVLM([answer("A road with a few cars.")]), DESCRIBE_FACTS, trace, "- a road")
    assert out == f"{UNGROUNDED_LABEL} A road with a few cars."


def test_describe_with_a_number_falls_back_to_the_captions_and_stays_labelled() -> None:
    trace = TraceCollector()
    out = compose(FakeVLM([answer("A road with 4 cars.")]), DESCRIBE_FACTS, trace, "- a road")
    assert out == f"{UNGROUNDED_LABEL} a road"
    assert verifier_counts(trace) == (0, 1)


def test_describe_without_captions_says_so() -> None:
    out = compose(FakeVLM(["x", "x"]), DESCRIBE_FACTS, TraceCollector())
    assert out.startswith(UNGROUNDED_LABEL) and "No shot captions" in out


# ---- verifier: ids are not quantities, and larger number words ---------------------------------

THREE_CARS = Facts(
    query_type="count",
    targets=["car"],
    count=3,
    tracks=[track_fact(1, 0.0, 3.2), track_fact(2, 1.0, 4.5), track_fact(3, 2.0, 5.0)],
)


def test_a_track_id_does_not_make_the_same_number_a_valid_count() -> None:
    assert verify("I counted 2 cars.", THREE_CARS, TOLERANCE)
    assert verify("There is 1 car.", THREE_CARS, TOLERANCE)
    assert verify("I counted 3 cars: tracks 1, 2 and 3, or #2.", THREE_CARS, TOLERANCE) == []


def test_a_track_reference_must_name_a_track_in_the_facts() -> None:
    assert verify("Track 7 is a car.", THREE_CARS, TOLERANCE) == ["track 7 is not in the facts"]
    assert verify("Tracks 1, 2 and 9.", THREE_CARS, TOLERANCE) == ["track 9 is not in the facts"]
    assert verify("Track 1, 2.0s to 3.2s", THREE_CARS, TOLERANCE) == []


@pytest.mark.parametrize(
    "text", ["There are thirty cars.", "about a hundred cars", "None were seen."]
)
def test_larger_number_words_and_none_are_checked(text: str) -> None:
    assert verify(text, THREE_CARS, TOLERANCE)
    zero = Facts(query_type="count", targets=["car"], count=0)
    assert verify("None were seen.", zero, TOLERANCE) == []


@pytest.mark.parametrize("text", ["Both cars are parked.", "A couple of cars.", "A dozen cars."])
def test_quantity_words_that_name_a_number_are_checked(text: str) -> None:
    assert verify(text, THREE_CARS, TOLERANCE)
    two = Facts(query_type="count", targets=["car"], count=2)
    assert verify("Both cars are parked.", two, TOLERANCE) == []
