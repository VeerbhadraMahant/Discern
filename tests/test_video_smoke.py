import pytest

from discern.eval.metrics import Counts
from discern.eval.synth_clips import ClipEntry, ClipObject
from discern.eval.video_smoke import (
    ClipRecord,
    aggregate,
    combine_metrics,
    count_question,
    fallback_events,
    ground_truth_frames,
    locate_label,
    locate_question,
    markdown_table,
    parse_count,
)
from discern.query.answer import VERIFY_NODE
from discern.trace import TraceEvent
from discern.vision.boxes import Box


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I counted 5 distinct car. track 3 (car) 0.0s to 8.0s.", 5),
        ("There are 12 cars in the video.", 12),
        ("There are three cars.", 3),
        ("Twenty-one people were seen.", 21),
        ("About thirty cars", 30),
        ("No car was found.", 0),
        ("none", 0),
        ("Tracks 1, 2 and 3 are cars; I counted 3.", 3),
        ("Track 4 appears at 1:05 and 2.5s.", None),
        ("The car appears.", None),
        ("Ungrounded (based on shot captions): 7 cars", 7),
    ],
)
def test_parse_count(text: str, expected: int | None) -> None:
    assert parse_count(text) == expected


def test_questions() -> None:
    assert count_question("car") == "How many car are there?"
    assert locate_question("person") == "Where is the first person?"


def test_locate_label_prefers_most_numerous_then_name() -> None:
    assert locate_label({"car": 3, "person": 5}) == "person"
    assert locate_label({"truck": 2, "bus": 2}) == "bus"
    assert locate_label({}) is None


def _entry() -> ClipEntry:
    return ClipEntry(
        clip="c/x.mp4",
        name="c",
        source_dataset="d",
        image_id="x",
        expected_counts={"car": 1},
        final_crop=Box(25, 20, 75, 60),
        fps=10.0,
        duration_seconds=0.3,
        n_frames=3,
        width=100,
        height=80,
        end_scale=0.5,
        min_visible_fraction=0.8,
        seed=1,
        gt_objects=[
            ClipObject(label="car", box=Box(30, 25, 50, 45)),
            ClipObject(label="car", box=Box(0, 0, 20, 20)),
            ClipObject(label="person", box=Box(40, 30, 60, 50)),
        ],
    )


def test_ground_truth_frames() -> None:
    gt = ground_truth_frames(_entry(), "car", [0, 2])
    assert [g.box for g in gt[0]] == [Box(30, 25, 50, 45), Box(0, 0, 20, 20)]
    assert [g.box for g in gt[2]] == [Box(10, 10, 50, 50)]  # final crop, mapped to the frame


def _event(node: str, fallback: bool) -> TraceEvent:
    return TraceEvent.model_validate(
        {
            "node": node,
            "started_at": "2026-01-01T00:00:00Z",
            "duration_ms": 1.0,
            "fallback_used": fallback,
        }
    )


def test_aggregate_hand_computed() -> None:
    a = ClipRecord(
        "c",
        "ds",
        "i",
        8.0,
        expected={"car": 4, "person": 2},
        predicted={"car": 4, "person": 5},
        box_counts=Counts(tp=3, fp=1, fn=1),
        tracks=[{"status": "accepted"}, {"status": "accepted"}, {"status": "rejected"}],
        events=[_event(VERIFY_NODE, False), _event("query_parse", True)],
        gpu_seconds_per_video_second=2.0,
    )
    b = ClipRecord(
        "c",
        "ds",
        "j",
        8.0,
        expected={"car": 1},
        predicted={"car": None},
        box_counts=Counts(tp=1, fp=0, fn=2),
        tracks=[{"status": "accepted"}],
        events=[_event(VERIFY_NODE, True), _event("adjudicate_track", True)],
        gpu_seconds_per_video_second=4.0,
    )
    failed = ClipRecord("c", "ds", "k", 8.0, expected={"car": 9}, error="boom")
    m = aggregate([a, b, failed])
    assert m["clips_ok"] == 2 and m["clips_failed"] == 1
    assert m["count_accuracy"] == pytest.approx(1 / 3)  # car=4 right; person and None wrong
    assert m["count_mae"] == pytest.approx((0 + 3 + 1) / 3)  # None counts as zero
    assert m["box_f1"] == pytest.approx(Counts(4, 1, 3).f1)
    assert m["verifier_pass_rate"] == pytest.approx(0.5)
    assert m["gpu_seconds_per_video_second"] == pytest.approx(3.0)
    assert m["tracks_per_clip"] == pytest.approx(1.5)  # accepted only: 2 and 1
    assert m["vlm_fallback_events"] == 2.0
    assert "wall_seconds_per_video_second" not in m


def test_fallback_events_exclude_the_verifier() -> None:
    events = [_event(VERIFY_NODE, True), _event("x", True), _event("y", False)]
    assert fallback_events(events) == 1


def test_aggregate_with_nothing_measured() -> None:
    assert aggregate([]) == {"clips_ok": 0.0, "clips_failed": 0.0}


def test_markdown_table() -> None:
    table = markdown_table({"d1": {"count_accuracy": 0.5, "clips_ok": 2.0}})
    lines = table.splitlines()
    assert lines[0].startswith("| dataset | count_accuracy |")
    assert lines[2].startswith("| d1 | 0.500 | - |")
    assert lines[2].endswith("| 2.000 | - |")


def test_combined_metrics_weight_by_measured_clips_and_sum_counts() -> None:
    per_dataset = {
        "a": {"clips_ok": 2.0, "clips_failed": 4.0,
              "count_accuracy": 1.0, "vlm_fallback_events": 1},
        "b": {"clips_ok": 6.0, "clips_failed": 0.0, "count_accuracy": 0.0, "box_f1": 0.5},
    }
    combined = combine_metrics(per_dataset)
    assert combined["count_accuracy"] == pytest.approx(0.25)  # 2 measured clips against 6
    assert combined["box_f1"] == 0.5  # only b has it
    assert combined["clips_ok"] == 8 and combined["clips_failed"] == 4
    assert combined["vlm_fallback_events"] == 1
