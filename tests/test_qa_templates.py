import pytest

from discern.config import load_settings
from discern.eval.qa_templates import (
    AnnotatedTrack,
    VideoAnnotation,
    bdd100k_to_annotation,
    flatten_chains,
    generate_chains,
    generate_questions,
    ground_truth_by_frame,
)
from discern.query.schemas import TimeRange
from discern.vision.boxes import Box

TH = load_settings("local_lite").thresholds.query


def _track(
    tid: int, label: str, frames: dict[int, float], x: float, y: float = 0
) -> AnnotatedTrack:
    return AnnotatedTrack(
        track_id=tid,
        label=label,
        boxes={f: Box(x, y, x + 10, y + 10) for f in frames},
        timestamps=frames,
    )


def _annotation() -> VideoAnnotation:
    return VideoAnnotation(
        video_id="v1",
        duration=10.0,
        width=100,
        height=50,
        tracks=(
            _track(1, "car", {0: 0.0, 5: 1.0, 10: 2.0}, x=0),  # left side, early
            _track(2, "car", {50: 6.0, 60: 7.0}, x=80),  # right side, late
            _track(3, "person", {0: 0.0, 5: 1.0}, x=40),
        ),
    )


def _by_text(questions: list, fragment: str):  # type: ignore[no-untyped-def]
    (match,) = [q for q in questions if fragment in q.text]
    return match


def test_single_questions_expected_answers() -> None:
    qs = generate_questions(_annotation(), TH, absent_labels=["train"])
    assert _by_text(qs, "How many cars are in the video").expected.count == 2
    # first half is [0, 5]: only the early car overlaps it
    assert _by_text(qs, "How many cars appear between").expected.count == 1
    assert _by_text(qs, "first car appear").expected.time == 0.0
    assert _by_text(qs, "Is there a person").expected.present is True
    assert _by_text(qs, "Is there a train").expected.present is False


def test_relation_question_time_ranges() -> None:
    qs = generate_questions(_annotation(), TH)
    car_left = _by_text(qs, "a car to the left of a person")
    assert car_left.expected.present is True
    assert car_left.expected.time_ranges == (TimeRange(start=0.0, end=1.0),)
    person_left = _by_text(qs, "a person to the left of a car")
    assert person_left.expected.present is False
    assert person_left.expected.time_ranges == ()


def test_question_ids_are_unique_and_generation_is_deterministic() -> None:
    a = generate_questions(_annotation(), TH)
    assert [q.id for q in a] == [q.id for q in generate_questions(_annotation(), TH)]
    assert len({q.id for q in a}) == len(a)


def test_chains_count_filter_locate() -> None:
    chains = {c.id: c for c in generate_chains(_annotation(), TH, region="left")}
    region = chains["v1:car:region"]
    assert [s.kind for s in region.steps] == ["count", "count", "locate"]
    assert [s.step for s in region.steps] == [0, 1, 2]
    assert all(s.chain_id == region.id for s in region.steps)
    assert region.steps[0].expected.count == 2
    assert region.steps[1].expected.count == 1  # only the left-side car
    boxes = region.steps[2].expected.boxes
    assert sorted(b.frame for b in boxes) == [0, 5, 10]
    timed = chains["v1:car:time"]
    assert timed.steps[1].expected.count == 1
    assert len(chains) == 4  # car and person, each with a region and a time chain


def test_flatten_chains_keeps_step_order() -> None:
    flat = flatten_chains(generate_chains(_annotation(), TH))
    assert [q.id for q in flat][:3] == ["v1:car:region:0", "v1:car:region:1", "v1:car:region:2"]


def test_ground_truth_by_frame_groups_boxes() -> None:
    chain = generate_chains(_annotation(), TH)[0]
    grouped = ground_truth_by_frame(chain.steps[2].expected.boxes)
    assert sorted(grouped) == [0, 5, 10]
    assert grouped[0][0].label == "car"


def _bdd_frame(index: int, labels: list[dict]) -> dict:  # type: ignore[type-arg]
    name = f"clip-{index:07d}.jpg"
    return {"name": name, "videoName": "clip", "frameIndex": index, "labels": labels}


def _obj(oid: str, category: str, x1: float = 0, y1: float = 0, x2: float = 10, y2: float = 10):  # type: ignore[no-untyped-def]
    return {"id": oid, "category": category, "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def test_bdd100k_loader() -> None:
    frames = [
        _bdd_frame(1, [_obj("b", "pedestrian", 5, 5, 15, 25), _obj("a", "car")]),
        _bdd_frame(0, [_obj("a", "car", 1, 1, 11, 11), _obj("z", "traffic sign")]),
        _bdd_frame(2, [_obj("a", "car", 2, 2, 12, 12)]),
    ]
    ann = bdd100k_to_annotation(frames, "clip", label_fps=5.0, video_fps=30.0)
    assert ann.duration == pytest.approx(3 / 5.0)
    # ids by first appearance in frame order: "a" is 1, "b" is 2; the sign is dropped
    car, person = ann.tracks
    assert (car.track_id, car.label) == (1, "car")
    assert (person.track_id, person.label) == (2, "person")
    assert sorted(car.boxes) == [0, 6, 12]  # frame numbers at 30 fps
    assert car.timestamps[6] == pytest.approx(0.2)
    assert car.boxes[0] == Box(1, 1, 11, 11)
    assert person.boxes[6] == Box(5, 5, 15, 25)


def test_bdd100k_loader_without_video_fps_keeps_label_indices() -> None:
    ann = bdd100k_to_annotation([_bdd_frame(3, [_obj("a", "bus")])], "clip")
    assert sorted(ann.tracks[0].boxes) == [3]


def test_bdd100k_loader_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        bdd100k_to_annotation([], "clip")
