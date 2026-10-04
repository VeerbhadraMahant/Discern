"""Experience-gated decision policy: code decides a node when the measured evidence is strong.

The memory holds measured F1 per option. Prompt-injected experience did not change the choices
of the 4-bit agent VLM, so when the evidence is decisive code makes the call (code computes, the
VLM judges); when it is not, `decide` returns None and the VLM decides as before.

Compared options: restorer = "none" vs the restorer mapped to the scene label; sr = "off" vs
"auto"; detector_set = the harvested detector sets (names joined by "+").
"""

from collections.abc import Callable, Collection

from pydantic import BaseModel, ConfigDict

from discern.agent.nodes.restorer_select import NONE, RESTORER_FOR_SCENE
from discern.agent.schemas import SceneProfile
from discern.config.settings import ExperienceThresholds
from discern.experience.aggregate import Memory
from discern.experience.injection import IMAGE_QUERY_TYPE
from discern.experience.retrieval import OptionSummary, Retrieval, retrieve
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


def _label(node: Node) -> str:
    return {"restorer": "restorer", "sr": "sr", "detector_set": "detectors"}[node]


def detector_decision(
    policy: ExperiencePolicy | None, k: int, available: Collection[str]
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
