"""Profile similarity and top-k retrieval with per-node recommendations (system-design 5.7)."""

from collections import defaultdict
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict

from discern.config.settings import ExperienceThresholds, SimilarityWeights
from discern.experience.aggregate import Memory, OptionStat
from discern.experience.schema import Node

NODES: tuple[Node, ...] = ("restorer", "sr", "detector_set")

# Ordinal scales per attribute, in profile-key order after the scene label.
_ORDER: dict[str, tuple[str, ...]] = {
    "illumination": ("dark", "dim", "normal", "bright"),
    "visibility": ("clear", "moderate", "poor"),
    "object_scale": ("small", "medium", "large"),
    "object_density": ("sparse", "moderate", "dense"),
}
_MIXED_VS_OTHER = 0.5  # "mixed" object scale has no ordinal position: half credit against others


def _parse(key: str) -> tuple[str, dict[str, str]]:
    label, *rest = key.split("|")
    return label, dict(zip(_ORDER, rest, strict=True))


def _match(attribute: str, a: str, b: str) -> float:
    if a == b:
        return 1.0
    if "mixed" in (a, b):
        return _MIXED_VS_OTHER
    order = _ORDER[attribute]
    return 1.0 - abs(order.index(a) - order.index(b)) / (len(order) - 1)


def profile_similarity(a: str, b: str, weights: SimilarityWeights) -> float:
    """Zero when scene labels differ, else a weighted ordinal match over the attributes (0 to 1)."""
    label_a, attrs_a = _parse(a)
    label_b, attrs_b = _parse(b)
    if label_a != label_b:
        return 0.0
    w = weights.model_dump()
    total = sum(w.values())
    if total <= 0:
        return 0.0
    return float(sum(w[k] * _match(k, attrs_a[k], attrs_b[k]) for k in _ORDER) / total)


class OptionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    option: str
    mean: float  # count-weighted mean over the retrieved profiles
    count: int


class Retrieval(BaseModel):
    model_config = ConfigDict(frozen=True)

    profiles: tuple[tuple[str, float], ...]  # (profile key, similarity), best first
    recommendations: dict[Node, tuple[OptionSummary, ...]]  # per node, best option first


def retrieve(
    memory: Memory,
    profile_key: str,
    query_type: str,
    settings: ExperienceThresholds,
) -> Retrieval:
    """Top `settings.top_k_profiles` profiles by similarity (zero-similarity ones are dropped)."""
    stats = [s for s in memory.stats if s.query_type == query_type]
    top = top_profiles({s.profile_key for s in stats}, profile_key, settings)
    chosen = {k for k, _ in top}
    return Retrieval(profiles=top, recommendations=_recommend(stats, chosen))


def top_profiles(
    keys: Iterable[str], profile_key: str, settings: ExperienceThresholds
) -> tuple[tuple[str, float], ...]:
    """The `settings.top_k_profiles` keys most similar to `profile_key`, best first; keys with
    zero similarity (another scene label) are dropped."""
    weights = settings.similarity_weights
    scored = [(k, profile_similarity(profile_key, k, weights)) for k in sorted(set(keys))]
    ranked = sorted((p for p in scored if p[1] > 0), key=lambda p: (-p[1], p[0]))
    return tuple(ranked[: settings.top_k_profiles])


def _recommend(
    stats: Sequence[OptionStat], chosen: set[str]
) -> dict[Node, tuple[OptionSummary, ...]]:
    sums: dict[Node, dict[str, list[float]]] = {n: defaultdict(lambda: [0.0, 0.0]) for n in NODES}
    for s in stats:
        if s.profile_key in chosen:
            acc = sums[s.node][s.option]
            acc[0] += s.mean * s.count
            acc[1] += s.count
    out: dict[Node, tuple[OptionSummary, ...]] = {}
    for node, options in sums.items():
        summaries = [
            OptionSummary(option=o, mean=total / n, count=int(n))
            for o, (total, n) in options.items()
        ]
        out[node] = tuple(sorted(summaries, key=lambda x: (-x.mean, x.option)))
    return out
