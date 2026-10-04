"""Pure helpers for the real-model video smoke evaluation (`scripts/eval_video_smoke.py`):
question wording, integer parsing of answers, per-clip records, aggregation into the gate's
metric names, and the markdown table. No model, torch or MLflow import."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from discern.eval.metrics import Counts
from discern.eval.synth_clips import ClipEntry, crop_at, visible_in_crop
from discern.eval.types import GroundTruthBox
from discern.eval.video_metrics import count_accuracy, count_mae, verifier_pass_rate
from discern.query.answer import VERIFY_NODE
from discern.trace import TraceEvent

METRIC_COLUMNS = (
    "count_accuracy",
    "count_mae",
    "box_f1",
    "verifier_pass_rate",
    "gpu_seconds_per_video_second",
    "tracks_per_clip",
    "vlm_fallback_events",
    "clips_ok",
    "clips_failed",
)

_UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split()
)
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_WORDS = {w: i for i, w in enumerate(_UNITS)} | {"none": 0, "no": 0}
_TRACK_REF = re.compile(
    r"(?:\btracks?\s*#?|#)\d+(?:\s*(?:,|and|&)\s*(?:and\s*)?#?\d+(?!\d|\.\d))*", re.IGNORECASE
)
_TIME = re.compile(
    r"\b\d+:\d{2}(?::\d{2})?(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|minutes?|mins?|s)\b",
    re.IGNORECASE,
)
_TOKEN = re.compile(
    r"\b(?P<digits>\d+)\b"
    r"|\b(?P<tens>" + "|".join(_TENS) + r")(?:[-\s](?P<unit>one|two|three|four|five|six|seven"
    r"|eight|nine))?\b"
    r"|\b(?P<word>" + "|".join(_WORDS) + r")\b",
    re.IGNORECASE,
)


def count_question(label: str) -> str:
    return f"How many {label} are there?"


def locate_question(label: str) -> str:
    return f"Where is the first {label}?"


def parse_count(answer: str) -> int | None:
    """The first quantity in an answer text: digits, or a number word (also "no" and "none").
    Track references ("track 3") and times ("2.5s", "1:05") are ignored. None when there is none."""
    text = _TIME.sub(" ", _TRACK_REF.sub(" ", answer))
    m = _TOKEN.search(text)
    if m is None:
        return None
    if m["digits"] is not None:
        return int(m["digits"])
    if m["tens"] is not None:
        value = 10 * (_TENS.index(m["tens"].lower()) + 2)
        return value + (_UNITS.index(m["unit"].lower()) if m["unit"] else 0)
    return _WORDS[m["word"].lower()]


def locate_label(expected: Mapping[str, int]) -> str | None:
    """Label asked about in the locate question: the most numerous, ties by name."""
    return min(expected, key=lambda k: (-expected[k], k), default=None)


def ground_truth_frames(
    entry: ClipEntry, label: str, frame_indices: Sequence[int]
) -> dict[int, list[GroundTruthBox]]:
    """Per-frame ground truth of `label` for the given decoded frame indices of a clip, rebuilt
    from the manifest entry (frame boxes are in the clip's pixels)."""
    gt = [GroundTruthBox(box=o.box, label=o.label) for o in entry.gt_objects]
    out: dict[int, list[GroundTruthBox]] = {}
    for f in frame_indices:
        crop = crop_at(entry.width, entry.height, entry.end_scale, f, entry.n_frames, entry.fps)
        seen = visible_in_crop(gt, crop, (entry.width, entry.height), entry.min_visible_fraction)
        out[f] = [GroundTruthBox(box=v.box, label=v.label) for v in seen if v.label == label]
    return out


@dataclass
class ClipRecord:
    clip: str
    dataset: str
    image_id: str
    duration: float
    expected: dict[str, int] = field(default_factory=dict)
    predicted: dict[str, int | None] = field(default_factory=dict)  # parsed from the answer text
    fact_counts: dict[str, int | None] = field(default_factory=dict)
    answers: dict[str, str] = field(default_factory=dict)
    locate_label: str | None = None
    locate_answer: str = ""
    box_counts: Counts | None = None  # None when the locate question was not scored
    tracks: list[dict[str, Any]] = field(default_factory=list)
    gpu_seconds_per_video_second: float | None = None
    wall_seconds_per_video_second: float | None = None
    events: list[TraceEvent] = field(default_factory=list)
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        bc = self.box_counts
        return {
            "clip": self.clip,
            "dataset": self.dataset,
            "image_id": self.image_id,
            "duration_seconds": self.duration,
            "expected": self.expected,
            "predicted": self.predicted,
            "fact_counts": self.fact_counts,
            "answers": self.answers,
            "locate_label": self.locate_label,
            "locate_answer": self.locate_answer,
            "box": None if bc is None else {"tp": bc.tp, "fp": bc.fp, "fn": bc.fn, "f1": bc.f1},
            "tracks": self.tracks,
            "gpu_seconds_per_video_second": self.gpu_seconds_per_video_second,
            "wall_seconds_per_video_second": self.wall_seconds_per_video_second,
            "fallback_events": fallback_events(self.events),
            "error": self.error,
        }


def fallback_events(events: Sequence[TraceEvent]) -> int:
    """Nodes that fell back to their deterministic default (VLM output unusable); the answer
    verifier is counted separately by `verifier_pass_rate`."""
    return sum(e.fallback_used for e in events if e.node != VERIFY_NODE)


def aggregate(records: Sequence[ClipRecord]) -> dict[str, float]:
    """Metrics over clips, named as the gate config names them where it has a name. Clips that
    failed are left out of every metric except `clips_failed`. A metric with nothing to measure
    is absent."""
    ok = [r for r in records if r.error is None]
    metrics: dict[str, float] = {
        "clips_ok": float(len(ok)),
        "clips_failed": float(len(records) - len(ok)),
    }
    predicted = [r.predicted.get(k) for r in ok for k in sorted(r.expected)]
    expected = [r.expected[k] for r in ok for k in sorted(r.expected)]
    if expected:
        metrics["count_accuracy"] = count_accuracy(predicted, expected)
        metrics["count_mae"] = count_mae(predicted, expected)
    total = Counts()
    scored = [r.box_counts for r in ok if r.box_counts is not None]
    for c in scored:
        total += c
    if scored:
        metrics["box_f1"] = total.f1
    events = [e for r in ok for e in r.events]
    rate = verifier_pass_rate(events)
    if rate is not None:
        metrics["verifier_pass_rate"] = rate
    gpu = [r.gpu_seconds_per_video_second for r in ok if r.gpu_seconds_per_video_second is not None]
    if gpu:
        metrics["gpu_seconds_per_video_second"] = fmean(gpu)
    wall = [
        r.wall_seconds_per_video_second
        for r in ok
        if r.wall_seconds_per_video_second is not None
    ]
    if wall:
        metrics["wall_seconds_per_video_second"] = fmean(wall)
    if ok:
        metrics["tracks_per_clip"] = fmean(
            sum(t["status"] == "accepted" for t in r.tracks) for r in ok
        )
        metrics["vlm_fallback_events"] = float(sum(fallback_events(r.events) for r in ok))
    return metrics


SUMMED_METRICS = frozenset({"clips_ok", "clips_failed", "vlm_fallback_events"})


def combine_metrics(per_dataset: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """One metric set over several clip sets. Counts are summed; every other metric is the mean
    of the clip sets that have it, weighted by their `clips_ok` (the clips it was measured on;
    failed clips are in no metric). Pooled metrics (box F1, verifier pass rate) are only
    approximated this way."""
    names = {n for m in per_dataset.values() for n in m}
    combined: dict[str, float] = {}
    for name in sorted(names):
        have = {d: m[name] for d, m in per_dataset.items() if name in m}
        if name in SUMMED_METRICS:
            combined[name] = sum(have.values())
            continue
        weights = {d: per_dataset[d].get("clips_ok", 0.0) for d in have}
        total = sum(weights.values())
        if total > 0:
            combined[name] = sum(v * weights[d] for d, v in have.items()) / total
    return combined


def markdown_table(by_dataset: Mapping[str, Mapping[str, float]]) -> str:
    """One row per dataset; a metric that was not measured shows as a dash."""
    head = "| dataset | " + " | ".join(METRIC_COLUMNS) + " |"
    rule = "|" + "---|" * (len(METRIC_COLUMNS) + 1)
    rows = [
        f"| {name} | "
        + " | ".join(f"{m[c]:.3f}" if c in m else "-" for c in METRIC_COLUMNS)
        + " |"
        for name, m in by_dataset.items()
    ]
    return "\n".join([head, rule, *rows])
