"""Experience-gated decision policy: code decides a node when the measured evidence is strong.

The memory holds measured F1 per option. Prompt-injected experience did not change the choices
of the 4-bit agent VLM, so when the evidence is decisive code makes the call (code computes, the
VLM judges); when it is not, `decide` returns None and the VLM decides as before.

Compared options: restorer = "none" vs the restorer mapped to the scene label; sr = "off" vs
"auto"; detector_set = the harvested detector sets (names joined by "+").
"""

from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from discern.agent.nodes.restorer_select import NONE, RESTORER_FOR_SCENE
from discern.agent.schemas import SceneProfile
from discern.config.settings import ExperienceThresholds
from discern.experience.aggregate import Memory
from discern.experience.injection import IMAGE_QUERY_TYPE
from discern.experience.retrieval import OptionSummary, Retrieval, retrieve, top_profiles
from discern.experience.schema import Node
from discern.trace import TraceCollector

MAPPED = "mapped"  # restorer decision: apply the restorer mapped to the scene label
SR_OFF = "off"
SR_AUTO = "auto"
DETECTOR_SEPARATOR = "+"
_PRECISION = 9  # decimals kept when comparing a margin, so 0.76 - 0.74 counts as 0.02



class PolicyDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    node: Node
    option: str  # "none" or "mapped"; "off" or "auto"; a detector set such as "a+b"
    rationale: str

    @property
    def detectors(self) -> list[str]:
        return self.option.split(DETECTOR_SEPARATOR)

class DecisionPolicy(Protocol):
    """What plan_image and detect_image need from a policy: the restorer the scene maps to and
    a per-node decision (None: the VLM decides that node). Implemented by the node-wise
    `ExperiencePolicy` and the `JointExperiencePolicy`."""

    mapped_restorer: str

    def decision(
        self, node: Node, accept: Callable[[str], bool] | None = None
    ) -> PolicyDecision | None: ...


class ExperiencePolicy:
    """Decisions for one scene profile from its retrieved experience."""

    def __init__(
        self, retrieval: Retrieval, mapped_restorer: str, settings: ExperienceThresholds
    ) -> None:
        self.mapped_restorer = mapped_restorer
        self._retrieval = retrieval
        self._settings = settings

    @classmethod
    def from_memory(
        cls,
        memory: Memory,
        profile: SceneProfile,
        settings: ExperienceThresholds,
        query_type: str = IMAGE_QUERY_TYPE,
    ) -> "ExperiencePolicy":
        found = retrieve(memory, profile.key, query_type, settings)
        return cls(found, RESTORER_FOR_SCENE[profile.scene_label], settings)

    def decide(self, node: Node, accept: Callable[[str], bool] | None = None) -> str | None:
        found = self.decision(node, accept)
        return None if found is None else found.option

    def decision(
        self, node: Node, accept: Callable[[str], bool] | None = None
    ) -> PolicyDecision | None:
        """The best option when it is backed by enough samples and beats the runner-up by the
        margin, else None. `accept` limits the detector-set options considered."""
        summaries = {o.option: o for o in self._retrieval.recommendations.get(node, ())}
        if node == "restorer":
            if self.mapped_restorer == NONE:
                return None  # nothing to compare: the scene maps to no restorer
            labels = {NONE: NONE, self.mapped_restorer: MAPPED}
            candidates = [(labels[k], summaries[k]) for k in labels if k in summaries]
        elif node == "sr":
            candidates = [(k, summaries[k]) for k in (SR_OFF, SR_AUTO) if k in summaries]
        else:
            candidates = [(k, s) for k, s in summaries.items() if accept is None or accept(k)]
        return self._pick(node, candidates)

    def _pick(
        self, node: Node, candidates: list[tuple[str, OptionSummary]]
    ) -> PolicyDecision | None:
        enough = [(o, s) for o, s in candidates if s.count >= self._settings.policy_min_count]
        if len(enough) < 2:
            return None
        ranked = sorted(enough, key=lambda c: (-c[1].mean, c[0]))
        (best, top), (_, second) = ranked[0], ranked[1]
        if round(top.mean - second.mean, _PRECISION) < self._settings.policy_margin:
            return None
        shown = self.mapped_restorer if best == MAPPED else best
        n = min(top.count, second.count)
        return PolicyDecision(
            node=node,
            option=best,
            rationale=(
                f"experience policy: {_label(node)} {shown} "
                f"({top.mean:.3f} vs {second.mean:.3f}, n={n})"
            ),
        )


class JointExperiencePolicy:
    """Decides restorer, SR and detector set together from raw per-configuration F1.

    Node-level values are maxima over the other nodes' options, so deciding the nodes one by one
    ignores how restoration, SR and the detector set interact. The memory's configuration stats
    hold the mean F1 of each whole configuration, pooled here (count-weighted) over the
    `top_k_profiles` most similar profiles.

    The baseline is the DEFAULT configuration: restorer none, SR off and the detector pair the
    system uses when no experience names one, the first K of `priority` (the deterministic
    priority order; detector_select makes no VLM call without experience). That is the
    experience-free choice that is cheap, measured and reproducible. The VLM's own unaided
    restorer and SR choices are not a fixed configuration that could be looked up, so they
    cannot be the baseline. Comparing against it asks the one question the memory can answer:
    does a non-default joint choice beat doing nothing special by the margin?

    A decision needs the winner and the default to have at least `policy_min_count` samples, and
    the winner's mean to beat the default's by at least `policy_margin`. Otherwise (including
    when the winner is the default itself) there is no decision and the VLM path runs as before.
    Only configurations over K distinct detectors from `priority` are candidates, so a decision
    never names an unavailable detector. Each node's `decision` returns its part of the one joint
    decision, all carrying the same rationale.
    """

    def __init__(
        self,
        pooled: dict[str, tuple[float, int]],  # configuration label -> (mean F1, count)
        mapped_restorer: str,
        priority: Sequence[str],
        k: int,
        settings: ExperienceThresholds,
    ) -> None:
        self.mapped_restorer = mapped_restorer
        self._decisions = self._decide(pooled, priority, k, settings)

    @classmethod
    def from_memory(
        cls,
        memory: Memory,
        profile: SceneProfile,
        settings: ExperienceThresholds,
        priority: Sequence[str],
        k: int,
        query_type: str = IMAGE_QUERY_TYPE,
    ) -> "JointExperiencePolicy":
        rows = [c for c in memory.config_stats if c.query_type == query_type]
        top = top_profiles({c.profile_key for c in rows}, profile.key, settings)
        chosen = {key for key, _ in top}
        sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for c in rows:
            if c.profile_key in chosen:
                sums[c.configuration][0] += c.mean * c.count
                sums[c.configuration][1] += c.count
        pooled = {label: (total / n, int(n)) for label, (total, n) in sums.items()}
        return cls(pooled, RESTORER_FOR_SCENE[profile.scene_label], priority, k, settings)

    def _decide(
        self,
        pooled: dict[str, tuple[float, int]],
        priority: Sequence[str],
        k: int,
        settings: ExperienceThresholds,
    ) -> dict[Node, PolicyDecision]:
        names = list(dict.fromkeys(priority))
        if len(names) < k:
            return {}
        default = f"{NONE}|{SR_OFF}|{DETECTOR_SEPARATOR.join(sorted(names[:k]))}"
        if default not in pooled or pooled[default][1] < settings.policy_min_count:
            return {}

        def usable(label: str) -> bool:
            parts = label.split("|")
            if len(parts) != 3:
                return False
            restorer, sr, detectors = parts
            chosen = detectors.split(DETECTOR_SEPARATOR)
            return (
                restorer in (NONE, self.mapped_restorer)
                and sr in (SR_OFF, SR_AUTO)
                and len(chosen) == k
                and len(set(chosen)) == k
                and set(chosen) <= set(names)
            )

        candidates = [
            (label, mean, n)
            for label, (mean, n) in pooled.items()
            if n >= settings.policy_min_count and usable(label)
        ]
        if not candidates:
            return {}
        label, mean, n = sorted(candidates, key=lambda c: (-c[1], c[0]))[0]
        base_mean, base_n = pooled[default]
        if label == default or round(mean - base_mean, _PRECISION) < settings.policy_margin:
            return {}
        restorer, sr, detectors = label.split("|")
        rationale = (
            f"experience joint policy: {label} {mean:.3f} vs default {base_mean:.3f}, "
            f"n={min(n, base_n)}"
        )
        options: dict[Node, str] = {
            "restorer": NONE if restorer == NONE else MAPPED,
            "sr": sr,
            "detector_set": detectors,
        }
        return {
            node: PolicyDecision(node=node, option=option, rationale=rationale)
            for node, option in options.items()
        }

    def decision(
        self, node: Node, accept: Callable[[str], bool] | None = None
    ) -> PolicyDecision | None:
        found = self._decisions.get(node)
        if found is not None and node == "detector_set" and accept is not None:
            return found if accept(found.option) else None
        return found


def build_policy(
    memory: Memory | None,
    profile: SceneProfile,
    settings: ExperienceThresholds,
    priority: Sequence[str],
    k: int,
) -> DecisionPolicy | None:
    """The policy `settings.policy_mode` names for `profile`; None without memory, or in joint
    mode when the memory has no configuration stats (the VLM then decides every node)."""
    if memory is None:
        return None
    if settings.policy_mode == "node":
        return ExperiencePolicy.from_memory(memory, profile, settings)
    if not memory.config_stats:
        return None
    return JointExperiencePolicy.from_memory(memory, profile, settings, priority, k)


def _label(node: Node) -> str:
    return {"restorer": "restorer", "sr": "sr", "detector_set": "detectors"}[node]


def detector_decision(
    policy: DecisionPolicy | None, k: int, available: Collection[str]
) -> PolicyDecision | None:
    """The policy's detector set when it names exactly `k` distinct detectors, all available."""

    def usable(option: str) -> bool:
        names = option.split(DETECTOR_SEPARATOR)
        return len(names) == k and len(set(names)) == k and all(n in available for n in names)

    return None if policy is None else policy.decision("detector_set", usable)


def record_decision(trace: TraceCollector, decision: PolicyDecision) -> None:
    """One trace event per policy decision (a decision made without a VLM call)."""
    with trace.span(f"experience_policy.{decision.node}") as span:
        span.input_summary = f"node={decision.node}"
        span.decision = decision.option
        span.rationale = decision.rationale
