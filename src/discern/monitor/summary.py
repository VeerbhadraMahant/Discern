"""Monitoring aggregates over saved traces and session records (system-design 5.8).

Pure functions over lists of records; no I/O.
"""

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from discern.config.settings import SimilarityWeights
from discern.experience.aggregate import Memory
from discern.experience.retrieval import profile_similarity
from discern.trace import TraceEvent


class SessionRecord(BaseModel):
    """What one answered request leaves behind for monitoring. Contains no media."""

    query_type: str
    profile_key: str | None = None  # scene profile of the upload; None when not profiled
    grounded: bool = True  # False when the answer had no grounded result
    events: list[TraceEvent] = Field(default_factory=list)


class MonitorSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests: int
    gpu_seconds_by_query_type: dict[str, float]
    fallback_rate_by_node: dict[str, float]  # fallback events / events, per node
    profile_distribution: dict[str, int]  # profile key -> uploads
    ungrounded_queries: int
    profiled_uploads: int
    drifted_uploads: int
    drift_share: float | None  # None when no upload was profiled


def drifted(profile_key: str, memory: Memory, weights: SimilarityWeights, threshold: float) -> bool:
    """True when no profile in the memory is at least `threshold` similar to `profile_key`."""
    keys = {s.profile_key for s in memory.stats}
    return not any(profile_similarity(profile_key, k, weights) >= threshold for k in keys)


def summarise(
    records: Sequence[SessionRecord],
    memory: Memory,
    weights: SimilarityWeights,
    drift_similarity: float,
) -> MonitorSummary:
    gpu_ms: dict[str, float] = defaultdict(float)
    events: dict[str, int] = defaultdict(int)
    fallbacks: dict[str, int] = defaultdict(int)
    profiles: dict[str, int] = defaultdict(int)
    drift_count = 0
    profiled = 0
    for r in records:
        gpu_ms[r.query_type] += sum(e.gpu_ms for e in r.events)
        for e in r.events:
            events[e.node] += 1
            fallbacks[e.node] += int(e.fallback_used)
        if r.profile_key is not None:
            profiled += 1
            profiles[r.profile_key] += 1
            drift_count += int(drifted(r.profile_key, memory, weights, drift_similarity))
    return MonitorSummary(
        requests=len(records),
        gpu_seconds_by_query_type={k: v / 1000.0 for k, v in sorted(gpu_ms.items())},
        fallback_rate_by_node={n: fallbacks[n] / c for n, c in sorted(events.items())},
        profile_distribution=dict(sorted(profiles.items())),
        ungrounded_queries=sum(1 for r in records if not r.grounded),
        profiled_uploads=profiled,
        drifted_uploads=drift_count,
        drift_share=drift_count / profiled if profiled else None,
    )


def render_markdown(summary: MonitorSummary) -> str:
    lines = ["# Monitoring", "", f"Requests: {summary.requests}", ""]
    lines += ["## GPU seconds per query type", "", "| Query type | GPU seconds |", "| --- | --- |"]
    lines += [f"| {k} | {v:.2f} |" for k, v in summary.gpu_seconds_by_query_type.items()]
    lines += ["", "## Fallback rate per node", "", "| Node | Fallback rate |", "| --- | --- |"]
    lines += [f"| {k} | {v:.0%} |" for k, v in summary.fallback_rate_by_node.items()]
    lines += ["", "## Scene profiles of uploads", "", "| Profile | Uploads |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in summary.profile_distribution.items()]
    drift = "n/a" if summary.drift_share is None else f"{summary.drift_share:.0%}"
    lines += [
        "",
        "## Drift",
        "",
        f"Uploads without a close experience match: {summary.drifted_uploads} of "
        f"{summary.profiled_uploads} ({drift}).",
        f"Queries with no grounded result: {summary.ungrounded_queries}.",
        "",
    ]
    return "\n".join(lines)
