"""Query executors (system-design 5.5): every Fact is computed in code from tracks. Detection and
tracking are never run here; the caller supplies a `TrackProvider` for scans that must run."""

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from discern.agent.nodes.attribute_check import attribute_check
from discern.config.settings import QueryThresholds, Settings
from discern.index.video_index import Segment, VideoIndex, retrieve
from discern.models.roles import VLM, Embedder
from discern.query.schemas import (
    ConversationState,
    Facts,
    Operation,
    PairFact,
    Predicate,
    QueryPlan,
    Region,
    TimeRange,
    TrackFact,
)
from discern.trace import TraceCollector
from discern.video.types import Track
from discern.vision.boxes import Box, intersection

# Detect and track `targets` and return the tracks (accepted and rejected). `segments` limits
# the scan to retrieved segments; None means the full video at the sampling stride.
TrackProvider = Callable[[Sequence[str], Sequence[Segment] | None], list[Track]]


@dataclass(frozen=True)
class Services:
    index: VideoIndex
    embedder: Embedder
    vlm: VLM
    trace: TraceCollector
    settings: Settings
    frame_size: tuple[int, int]  # (width, height) of original frames
    detect: TrackProvider


@dataclass
class Execution:
    facts: Facts
    tracks: list[Track]  # the result set's accepted tracks
    context: str = ""  # shot captions for ungrounded answers


# ---- geometry (pure) --------------------------------------------------------------------------


def _centre(b: Box) -> tuple[float, float]:
    return (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2


def relation_holds(predicate: Predicate, a: Box, b: Box, th: QueryThresholds) -> bool:
    """Whether `a` stands in `predicate` to `b`, from boxes in the same frame."""
    (ax, ay), (bx, by) = _centre(a), _centre(b)
    if predicate == "overlapping":
        smaller = min(a.area, b.area)
        return bool(smaller > 0 and intersection(a, b).area / smaller >= th.overlap_ratio)
    if predicate == "near":
        diagonal = (math.hypot(a.width, a.height) + math.hypot(b.width, b.height)) / 2
        return math.hypot(ax - bx, ay - by) <= th.near_ratio * diagonal
    mean_w, mean_h = (a.width + b.width) / 2, (a.height + b.height) / 2
    if predicate == "left_of":
        return bx - ax >= th.direction_ratio * mean_w
    if predicate == "right_of":
        return ax - bx >= th.direction_ratio * mean_w
    if predicate == "above":
        return by - ay >= th.direction_ratio * mean_h
    return ay - by >= th.direction_ratio * mean_h  # below


def _in_region(box: Box, region: Region, frame_size: tuple[int, int]) -> bool:
    cx, cy = _centre(box)
    width, height = frame_size
    inside: dict[Region, bool] = {
        "left": cx < width / 2,
        "right": cx >= width / 2,
        "top": cy < height / 2,
        "bottom": cy >= height / 2,
    }
    return inside[region]


# ---- track helpers ----------------------------------------------------------------------------


def _accepted(tracks: Sequence[Track]) -> list[Track]:
    return [t for t in tracks if t.status == "accepted"]


def _overlaps(track: Track, window: TimeRange) -> bool:
    return track.t_start <= window.end and track.t_end >= window.start


def _track_fact(track: Track) -> TrackFact:
    return TrackFact(
        track_id=track.id,
        label=track.label,
        t_start=track.t_start,
        t_end=track.t_end,
        boxes=dict(track.frames),
    )


def _filter_attribute(
    tracks: Sequence[Track],
    attribute: str,
    value: str,
    services: Services,
    state: ConversationState,
) -> list[Track]:
    """Keep tracks whose classified attribute equals `value`. Classifications are cached per
    (track, attribute) for the whole conversation."""
    attribute, wanted = attribute.strip().lower(), value.strip().lower()
    kept = []
    for track in tracks:
        key = (track.id, attribute)
        if key not in state.attribute_cache:
            found = attribute_check(services.vlm, services.trace, track, attribute)
            if found == "unknown":  # not cached: a later attempt may succeed
                continue
            state.attribute_cache[key] = found
        if state.attribute_cache[key] == wanted:
            kept.append(track)
    return kept


def _filter_region(
    tracks: Sequence[Track], region: Region, services: Services
) -> list[Track]:
    need = services.settings.thresholds.query.region_min_fraction
    return [
        t
        for t in tracks
        if sum(_in_region(b, region, services.frame_size) for b in t.frames.values())
        / len(t.frames)
        >= need
    ]


def _scan(
    targets: Sequence[str], services: Services, use_retrieval: bool
) -> list[Track]:
    """Accepted tracks for `targets`. With retrieval, scan the best segments first and fall back
    to a full scan when they yield nothing (system-design 5.4)."""
    if use_retrieval:
        segments = retrieve(
            services.index,
            services.embedder,
            ", ".join(targets),
            services.settings.thresholds.index.top_segments,
            services.settings,
        )
        if segments:
            found = _accepted(services.detect(targets, segments))
            if found:
                return found
            with services.trace.span("scan.fallback") as span:
                span.input_summary = f"targets={list(targets)}, segments={len(segments)}"
                span.decision = "retrieved segments had no accepted tracks, scanning full video"
    return _accepted(services.detect(targets, None))


def _candidates(
    plan: QueryPlan, services: Services, state: ConversationState, use_retrieval: bool
) -> tuple[list[str], list[Track]]:
    """Tracks the plan applies to: the source result set's when given (no detection), else a
    scan. Attribute constraints and the time range are applied."""
    if plan.source_result_set:
        source = state.result_sets[plan.source_result_set]
        targets, tracks = plan.targets or source.facts.targets, list(source.tracks)
    else:
        targets, tracks = plan.targets, _scan(plan.targets, services, use_retrieval)
    for constraint in plan.attributes:
        tracks = _filter_attribute(tracks, constraint.attribute, constraint.value, services, state)
    if plan.time_range:
        window = plan.time_range
        tracks = [t for t in tracks if _overlaps(t, window)]
    return list(targets), tracks


# ---- executors --------------------------------------------------------------------------------


def _locate(plan: QueryPlan, services: Services, state: ConversationState) -> Execution:
    targets, tracks = _candidates(plan, services, state, use_retrieval=True)
    facts = Facts(
        query_type="locate", targets=targets, count=len(tracks),
        tracks=[_track_fact(t) for t in tracks],
    )
    return Execution(facts, tracks)


def _count(plan: QueryPlan, services: Services, state: ConversationState) -> Execution:
    targets, tracks = _candidates(plan, services, state, use_retrieval=False)
    facts = Facts(
        query_type="count", targets=targets, count=len(tracks),
        tracks=[_track_fact(t) for t in tracks],
    )
    return Execution(facts, tracks)


def _temporal(plan: QueryPlan, services: Services, state: ConversationState) -> Execution:
    targets, tracks = _candidates(plan, services, state, use_retrieval=True)
    facts = Facts(
        query_type="temporal", targets=targets, count=len(tracks),
        tracks=[_track_fact(t) for t in tracks],
        time_ranges=[TimeRange(start=t.t_start, end=t.t_end) for t in tracks],
    )
    return Execution(facts, tracks)


def _holding_runs(
    a: Track, b: Track, predicate: Predicate, th: QueryThresholds
) -> list[TimeRange]:
    """Time ranges over which `a` stands in `predicate` to `b`, from the sampled frames both
    tracks were observed in. A run is consecutive such frames."""
    shared = sorted(set(a.frames) & set(b.frames))
    runs: list[TimeRange] = []
    start: float | None = None
    last = 0.0
    for index in shared:
        if relation_holds(predicate, a.frames[index], b.frames[index], th):
            if start is None:
                start = a.timestamps[index]
            last = a.timestamps[index]
        elif start is not None:
            runs.append(TimeRange(start=start, end=last))
            start = None
    if start is not None:
        runs.append(TimeRange(start=start, end=last))
    return runs


def _relation(plan: QueryPlan, services: Services, state: ConversationState) -> Execution:
    """The first relation of the plan: pairs of distinct tracks where it holds, and when."""
    rel = plan.relations[0]
    th = services.settings.thresholds.query

    def constrained(tracks: list[Track], target: str) -> list[Track]:
        for c in plan.attributes:  # attribute constraints name the entity they describe
            if c.target.strip().lower() == target.strip().lower():
                tracks = _filter_attribute(tracks, c.attribute, c.value, services, state)
        return tracks

    subjects = constrained(_scan([rel.subject], services, use_retrieval=False), rel.subject)
    objects = subjects
    if rel.object != rel.subject:
        objects = constrained(_scan([rel.object], services, use_retrieval=False), rel.object)
    symmetric = rel.subject == rel.object and rel.predicate in ("near", "overlapping")
    pairs: list[PairFact] = []
    involved: dict[int, Track] = {}
    for a, b in itertools.product(subjects, objects):
        if a.id == b.id or (symmetric and a.id > b.id):  # "A near B" is "B near A": one pair
            continue
        runs = _holding_runs(a, b, rel.predicate, th)
        if plan.time_range:
            window = plan.time_range
            runs = [r for r in runs if r.start <= window.end and r.end >= window.start]
        if runs:
            pairs.append(PairFact(subject_track=a.id, object_track=b.id, time_ranges=runs))
            involved |= {a.id: a, b.id: b}
    facts = Facts(
        query_type="relation",
        targets=[rel.subject, rel.object],
        count=len(pairs),
        tracks=[_track_fact(t) for t in involved.values()],
        time_ranges=[r for p in pairs for r in p.time_ranges],
        pairs=pairs,
        relation=f"{rel.subject} {rel.predicate} {rel.object}",
    )
    return Execution(facts, list(involved.values()))


def _describe(plan: QueryPlan, services: Services, state: ConversationState) -> Execution:
    """Ungrounded: the answer is phrased from shot captions and Facts stay empty."""
    window = plan.time_range
    lines = [
        f"- {services.index.captions[s.id]}"
        for s in services.index.shots
        if services.index.captions.get(s.id)
        and (window is None or (s.t_start <= window.end and s.t_end >= window.start))
    ]
    return Execution(Facts(query_type="describe", grounded=False), [], "\n".join(lines))


def _apply(
    op: Operation, tracks: list[Track], services: Services, state: ConversationState
) -> list[Track]:
    if op.kind == "filter_attribute":
        assert op.attribute is not None and op.value is not None
        return _filter_attribute(tracks, op.attribute, op.value, services, state)
    if op.kind == "filter_time":
        assert op.time_range is not None
        window = op.time_range
        return [t for t in tracks if _overlaps(t, window)]
    if op.kind == "filter_region":
        assert op.region is not None
        return _filter_region(tracks, op.region, services)
    return tracks  # count, show and compare do not change the set


def _refine(plan: QueryPlan, services: Services, state: ConversationState) -> Execution:
    """Operate on an existing result set; no detection runs."""
    assert plan.source_result_set is not None
    source = state.result_sets[plan.source_result_set]
    tracks = list(source.tracks)
    compared: dict[str, int] = {}
    for op in plan.operations:
        tracks = _apply(op, tracks, services, state)
        if op.kind == "compare":
            assert op.value is not None
            compared = {source.id: len(tracks), op.value: len(state.result_sets[op.value].tracks)}
    facts = Facts(
        query_type="refine",
        targets=source.facts.targets,
        count=len(tracks),
        tracks=[_track_fact(t) for t in tracks],
        compared=compared,
    )
    return Execution(facts, tracks)


_EXECUTORS: dict[str, Callable[[QueryPlan, Services, ConversationState], Execution]] = {
    "locate": _locate,
    "count": _count,
    "temporal": _temporal,
    "relation": _relation,
    "describe": _describe,
    "refine": _refine,
}


def execute(plan: QueryPlan, services: Services, state: ConversationState) -> Execution:
    """Run the executor for the plan's query type and trace what it computed."""
    with services.trace.span(f"execute.{plan.query_type}") as span:
        span.input_summary = (
            f"targets={plan.targets}, source={plan.source_result_set}, "
            f"operations={[o.kind for o in plan.operations]}"
        )
        result = _EXECUTORS[plan.query_type](plan, services, state)
        span.decision = f"count={result.facts.count}, tracks={len(result.tracks)}"
    return result
