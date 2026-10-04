"""Offline harvest for one labelled image (system-design 5.7 steps 2 to 5).

All inputs (cached detector outputs per image variant, the image, ground truth) are passed in, so
no model runs here. Every configuration is a recombination of cached outputs, scored by cheap
fusion (grouping without adjudication) and F1@0.5. The top configurations are re-scored by a
caller-supplied `confirm` (full adjudication) and the confirmed value replaces the cheap one.
"""

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations

from pydantic import BaseModel, ConfigDict

from discern.agent.schemas import SceneProfile
from discern.config.settings import GroupingThresholds
from discern.eval.metrics import match_image
from discern.eval.types import GroundTruthBox
from discern.experience.schema import NODE_METRIC, ExperienceRecord, Node, RecordNode
from discern.models.roles import Detection, Image
from discern.vision.grouping import group_detections

NONE = "none"
SR_OFF = "off"
SR_AUTO = "auto"

VariantKey = tuple[str, str]  # (restorer option, sr option)
CachedOutputs = Mapping[VariantKey, Mapping[str, Sequence[Detection]]]  # variant -> detector -> out


class Configuration(BaseModel):
    model_config = ConfigDict(frozen=True)

    restorer: str
    sr: str
    detectors: tuple[str, ...]

    @property
    def variant(self) -> VariantKey:
        return (self.restorer, self.sr)

    def option(self, node: Node) -> str:
        if node == "restorer":
            return self.restorer
        if node == "sr":
            return self.sr
        return "+".join(self.detectors)

    @property
    def label(self) -> str:
        return f"{self.restorer}|{self.sr}|{self.option('detector_set')}"


class ConfigScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    config: Configuration
    fused_f1: float
    confirmed_f1: float | None = None

    @property
    def f1(self) -> float:
        return self.fused_f1 if self.confirmed_f1 is None else self.confirmed_f1


def enumerate_configurations(
    mapped_restorer: str, detector_pool: Sequence[str]
) -> list[Configuration]:
    """Restorer in {none, mapped}, SR in {off, auto}, detector set = all singles and pairs."""
    restorers = [NONE] if mapped_restorer == NONE else [NONE, mapped_restorer]
    pool = sorted(set(detector_pool))
    sets = [(d,) for d in pool] + list(combinations(pool, 2))
    return [
        Configuration(restorer=r, sr=s, detectors=ds)
        for r in restorers
        for s in (SR_OFF, SR_AUTO)
        for ds in sets
    ]


def _floor(min_score: float | Mapping[str, float], detector: str) -> float:
    """Score floor for one detector: a single value for all, or a per-detector mapping."""
    return min_score if isinstance(min_score, float | int) else min_score.get(detector, 0.0)


def detector_floors(
    min_score: float | Mapping[str, float], detectors: Sequence[str]
) -> dict[str, float]:
    """Score floor of each of `detectors`, the operating thresholds of the adjudicated path."""
    return {name: _floor(min_score, name) for name in detectors}


def fused_f1(
    config: Configuration,
    image: Image,
    ground_truth: Sequence[GroundTruthBox],
    outputs: CachedOutputs,
    grouping: GroupingThresholds,
    min_score: float | Mapping[str, float] = 0.0,
) -> float:
    """Cheap fusion: pool the set's detections, group, keep each group's anchor, F1@0.5."""
    try:
        per_detector = outputs[config.variant]
        pooled = [
            d
            for name in config.detectors
            for d in per_detector[name]
            if d.score >= _floor(min_score, name)
        ]
    except KeyError as e:
        raise ValueError(f"no cached detector output for {config.label}: {e}") from e
    anchors = [g.anchor for g in group_detections(image, pooled, grouping)]
    return match_image(anchors, ground_truth).f1


def score_image(
    configs: Sequence[Configuration],
    image: Image,
    ground_truth: Sequence[GroundTruthBox],
    outputs: CachedOutputs,
    grouping: GroupingThresholds,
    confirm: Callable[[Configuration], float],
    confirm_top: int,
    min_score: float | Mapping[str, float] = 0.0,
) -> list[ConfigScore]:
    """Score all configurations cheaply, then confirm the best `confirm_top` (ties: input order)."""
    scores = [
        ConfigScore(
            config=c, fused_f1=fused_f1(c, image, ground_truth, outputs, grouping, min_score)
        )
        for c in configs
    ]
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i].fused_f1, i))
    for i in ranked[:confirm_top]:
        scores[i] = scores[i].model_copy(update={"confirmed_f1": confirm(scores[i].config)})
    return scores


def node_values(scores: Sequence[ConfigScore], node: Node) -> dict[str, float]:
    """Best F1 achievable given each option of `node` (max over the other nodes' options).

    Confirmed scores replace cheap ones for the configurations that were confirmed.
    """
    best: dict[str, float] = {}
    for s in scores:
        opt = s.config.option(node)
        best[opt] = max(best.get(opt, 0.0), s.f1)
    return best


def harvest_image(
    sample_id: str,
    profile: SceneProfile,
    query_type: str,
    image: Image,
    ground_truth: Sequence[GroundTruthBox],
    outputs: CachedOutputs,
    mapped_restorer: str,
    detector_pool: Sequence[str],
    grouping: GroupingThresholds,
    confirm: Callable[[Configuration], float],
    confirm_top: int,
    memory_version: str,
    min_score: float | Mapping[str, float] = 0.0,
) -> list[ExperienceRecord]:
    """Node-level records plus one raw record per configuration, for one labelled image."""
    configs = enumerate_configurations(mapped_restorer, detector_pool)
    scores = score_image(
        configs, image, ground_truth, outputs, grouping, confirm, confirm_top, min_score
    )

    def record(node: RecordNode, option: str, name: str, value: float) -> ExperienceRecord:
        return ExperienceRecord(
            profile_key=profile.key,
            query_type=query_type,
            node=node,
            option=option,
            metric_name=name,
            metric_value=value,
            sample_id=sample_id,
            source="benchmark",
            memory_version=memory_version,
        )

    records: list[ExperienceRecord] = []
    nodes: tuple[Node, ...] = ("restorer", "sr", "detector_set")
    for node in nodes:
        for option, value in sorted(node_values(scores, node).items()):
            records.append(record(node, option, NODE_METRIC, value))
    for s in scores:
        name = "f1_fused" if s.confirmed_f1 is None else "f1_adjudicated"
        records.append(record("configuration", s.config.label, name, s.f1))
    return records
