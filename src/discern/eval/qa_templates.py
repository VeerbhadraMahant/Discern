"""Automatically checkable video questions from ground-truth tracks (system-design 10.2).

`VideoAnnotation` is the ground truth. Expected answers are computed with the same geometry the
query executors use (`relation_holds`, region membership, time overlap), so a perfect system
scores perfectly. Questions are deterministic: same annotation, same questions, same order.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from discern.config.settings import QueryThresholds
from discern.eval.datasets import BDD_LABELS
from discern.eval.types import GroundTruthBox
from discern.query.executors import _in_region, relation_holds
from discern.query.schemas import Predicate, Region, TimeRange
from discern.vision.boxes import Box

QuestionKind = Literal["count", "presence", "first_appearance", "relation", "locate"]

_PREDICATE_TEXT: dict[Predicate, str] = {
    "near": "near",
    "left_of": "to the left of",
    "right_of": "to the right of",
    "above": "above",
    "below": "below",
    "overlapping": "overlapping",
}


class AnnotatedTrack(BaseModel):
    model_config = ConfigDict(frozen=True)

    track_id: int
    label: str
    boxes: dict[int, Box]  # frame index -> box, original-frame pixels
    timestamps: dict[int, float]  # frame index -> seconds

    @property
    def t_start(self) -> float:
        return min(self.timestamps.values())

    @property
    def t_end(self) -> float:
        return max(self.timestamps.values())


class VideoAnnotation(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    duration: float
    width: int
    height: int
    tracks: tuple[AnnotatedTrack, ...]

    @property
    def labels(self) -> list[str]:
        return sorted({t.label for t in self.tracks})


class FrameBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame: int
    label: str
    box: Box


class Expected(BaseModel):
    """The checkable answer. Which fields are set depends on the question kind."""

    model_config = ConfigDict(frozen=True)

    count: int | None = None
    present: bool | None = None
    time: float | None = None  # first appearance
    time_ranges: tuple[TimeRange, ...] = ()
    boxes: tuple[FrameBox, ...] = ()  # locate: every ground-truth box of the answer tracks


class VideoQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    video_id: str
    kind: QuestionKind
    text: str
    expected: Expected
    chain_id: str | None = None
    step: int = 0


class QuestionChain(BaseModel):
    """A scripted multi-turn sequence. Steps must be asked in order in one conversation."""

    model_config = ConfigDict(frozen=True)

    id: str
    video_id: str
    steps: tuple[VideoQuestion, ...] = Field(min_length=1)


def flatten_chains(chains: Sequence[QuestionChain]) -> list[VideoQuestion]:
    return [q for c in chains for q in c.steps]


# ---- ground-truth helpers ---------------------------------------------------------------------


def _of(ann: VideoAnnotation, label: str) -> list[AnnotatedTrack]:
    return [t for t in ann.tracks if t.label == label]


def _overlaps(track: AnnotatedTrack, window: TimeRange) -> bool:
    return track.t_start <= window.end and track.t_end >= window.start


def _in_window(tracks: Sequence[AnnotatedTrack], window: TimeRange) -> list[AnnotatedTrack]:
    return [t for t in tracks if _overlaps(t, window)]


def _in_region_tracks(
    ann: VideoAnnotation, tracks: Sequence[AnnotatedTrack], region: Region, th: QueryThresholds
) -> list[AnnotatedTrack]:
    size = (ann.width, ann.height)
    return [
        t
        for t in tracks
        if sum(_in_region(b, region, size) for b in t.boxes.values()) / len(t.boxes)
        >= th.region_min_fraction
    ]


def _frame_boxes(tracks: Sequence[AnnotatedTrack]) -> tuple[FrameBox, ...]:
    return tuple(
        FrameBox(frame=f, label=t.label, box=b)
        for t in tracks
        for f, b in sorted(t.boxes.items())
    )


def ground_truth_by_frame(boxes: Sequence[FrameBox]) -> dict[int, list[GroundTruthBox]]:
    """Group expected boxes by frame, in the shape `box_f1_at_frames` takes."""
    out: dict[int, list[GroundTruthBox]] = {}
    for fb in boxes:
        out.setdefault(fb.frame, []).append(GroundTruthBox(box=fb.box, label=fb.label))
    return out


def _holding_ranges(
    a: AnnotatedTrack, b: AnnotatedTrack, predicate: Predicate, th: QueryThresholds
) -> list[TimeRange]:
    """Runs of consecutive shared frames over which `a` stands in `predicate` to `b`."""
    runs: list[TimeRange] = []
    start: float | None = None
    last = 0.0
    for f in sorted(set(a.boxes) & set(b.boxes)):
        if relation_holds(predicate, a.boxes[f], b.boxes[f], th):
            start = a.timestamps[f] if start is None else start
            last = a.timestamps[f]
        elif start is not None:
            runs.append(TimeRange(start=start, end=last))
            start = None
    if start is not None:
        runs.append(TimeRange(start=start, end=last))
    return runs


def _plural(label: str) -> str:
    return label + "s" if not label.endswith("s") else label


# ---- single questions -------------------------------------------------------------------------


def generate_questions(
    ann: VideoAnnotation,
    th: QueryThresholds,
    absent_labels: Sequence[str] = (),
    predicates: Sequence[Predicate] = ("left_of",),
) -> list[VideoQuestion]:
    """Count (whole video and first half), first appearance, presence (present labels, plus
    `absent_labels` as negatives) and relation questions (every ordered pair of distinct
    labels for each predicate) for one annotated video."""
    vid, labels = ann.video_id, ann.labels
    first_half = TimeRange(start=0.0, end=ann.duration / 2)
    questions: list[VideoQuestion] = []

    def add(kind: QuestionKind, text: str, expected: Expected) -> None:
        questions.append(
            VideoQuestion(
                id=f"{vid}:{len(questions)}", video_id=vid, kind=kind, text=text, expected=expected
            )
        )

    for label in labels:
        tracks = _of(ann, label)
        add("count", f"How many {_plural(label)} are in the video?", Expected(count=len(tracks)))
        windowed = _in_window(tracks, first_half)
        add(
            "count",
            f"How many {_plural(label)} appear between {first_half.start:g} and "
            f"{first_half.end:g} seconds?",
            Expected(count=len(windowed)),
        )
        add(
            "first_appearance",
            f"At what time does the first {label} appear?",
            Expected(time=min(t.t_start for t in tracks)),
        )
        add("presence", f"Is there a {label} in the video?", Expected(present=True))
    for label in absent_labels:
        present = bool(_of(ann, label))
        add("presence", f"Is there a {label} in the video?", Expected(present=present))
    for subject in labels:
        for obj in labels:
            if subject == obj:
                continue
            for predicate in predicates:
                runs = [
                    r
                    for a in _of(ann, subject)
                    for b in _of(ann, obj)
                    for r in _holding_ranges(a, b, predicate, th)
                ]
                add(
                    "relation",
                    f"When is a {subject} {_PREDICATE_TEXT[predicate]} a {obj}?",
                    Expected(present=bool(runs), time_ranges=tuple(runs)),
                )
    return questions


# ---- follow-up chains -------------------------------------------------------------------------


def generate_chains(
    ann: VideoAnnotation, th: QueryThresholds, region: Region = "left"
) -> list[QuestionChain]:
    """Per label with at least one track, two chains of count, then filter, then locate:
    one filters by frame region, the other by the first half of the video. Each step's
    expected answer is computed on the tracks left after the previous steps."""
    vid = ann.video_id
    first_half = TimeRange(start=0.0, end=ann.duration / 2)
    chains: list[QuestionChain] = []
    for label in ann.labels:
        plural = _plural(label)
        everything = _of(ann, label)
        filters: list[tuple[str, str, list[AnnotatedTrack]]] = [
            (
                "region",
                f"Of those, how many are on the {region} side of the frame?",
                _in_region_tracks(ann, everything, region, th),
            ),
            (
                "time",
                f"Of those, how many appear between {first_half.start:g} and "
                f"{first_half.end:g} seconds?",
                _in_window(everything, first_half),
            ),
        ]
        for name, filter_text, kept in filters:
            chain_id = f"{vid}:{label}:{name}"
            texts: list[tuple[QuestionKind, str, Expected]] = [
                ("count", f"How many {plural} are in the video?", Expected(count=len(everything))),
                ("count", filter_text, Expected(count=len(kept))),
                ("locate", "Where are they?", Expected(boxes=_frame_boxes(kept))),
            ]
            chains.append(
                QuestionChain(
                    id=chain_id,
                    video_id=vid,
                    steps=tuple(
                        VideoQuestion(
                            id=f"{chain_id}:{i}",
                            video_id=vid,
                            kind=kind,
                            text=text,
                            expected=expected,
                            chain_id=chain_id,
                            step=i,
                        )
                        for i, (kind, text, expected) in enumerate(texts)
                    ),
                )
            )
    return chains


# ---- BDD100K loader ---------------------------------------------------------------------------


def bdd100k_to_annotation(
    frames: Sequence[Mapping[str, Any]],
    video_id: str,
    width: int = 1280,
    height: int = 720,
    label_fps: float = 5.0,
    video_fps: float | None = None,
) -> VideoAnnotation:
    """Convert one parsed BDD100K MOT label file (a list of per-frame dicts with `frameIndex`
    and `labels`, each label having `id`, `category` and `box2d`) to a `VideoAnnotation`.

    Labelled frames are `label_fps` apart. Frame keys are mapped to the original video's frame
    numbers when `video_fps` is given (default: no mapping). Categories outside the shared
    vocabulary are dropped. Track ids are assigned in order of first appearance.
    """
    if not frames:
        raise ValueError("no frames to convert")
    ids: dict[str, int] = {}
    labels: dict[int, str] = {}
    boxes: dict[int, dict[int, Box]] = {}
    stamps: dict[int, dict[int, float]] = {}
    for frame in sorted(frames, key=lambda f: int(f["frameIndex"])):
        seconds = int(frame["frameIndex"]) / label_fps
        key = int(frame["frameIndex"]) if video_fps is None else round(seconds * video_fps)
        for obj in frame.get("labels") or []:
            label = BDD_LABELS.get(obj["category"])
            box2d = obj.get("box2d")
            if label is None or box2d is None:
                continue
            tid = ids.setdefault(str(obj["id"]), len(ids) + 1)
            labels.setdefault(tid, label)
            boxes.setdefault(tid, {})[key] = Box(
                float(box2d["x1"]), float(box2d["y1"]), float(box2d["x2"]), float(box2d["y2"])
            )
            stamps.setdefault(tid, {})[key] = seconds
    last = max(int(f["frameIndex"]) for f in frames)
    return VideoAnnotation(
        video_id=video_id,
        duration=(last + 1) / label_fps,
        width=width,
        height=height,
        tracks=tuple(
            AnnotatedTrack(
                track_id=tid, label=labels[tid], boxes=boxes[tid], timestamps=stamps[tid]
            )
            for tid in sorted(boxes)
        ),
    )
