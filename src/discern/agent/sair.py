"""SAIR orchestration: perception, restorer and image selection, SR selection."""

import logging
from collections.abc import Callable, Mapping
from typing import Literal

from discern.agent.nodes.image_select import image_select
from discern.agent.nodes.perception import perception
from discern.agent.nodes.restorer_select import NONE, restorer_select
from discern.agent.nodes.sr_select import required_factor, sr_select
from discern.agent.schemas import SceneProfile, ShotPlan
from discern.config.settings import Settings, load_settings
from discern.experience.policy import MAPPED, SR_AUTO, ExperiencePolicy, record_decision
from discern.models.roles import VLM, Image, Restorer
from discern.trace import TraceCollector

logger = logging.getLogger(__name__)

# Experience text for the restorer and SR prompts: fixed, or computed from the perceived profile.
Experience = str | Callable[[SceneProfile], str]
# Experience-gated decisions: fixed, or computed from the perceived profile (None = no policy).
Policy = ExperiencePolicy | Callable[[SceneProfile], ExperiencePolicy | None] | None


def plan_image(
    vlm: VLM,
    trace: TraceCollector,
    image: Image,
    restorers: Mapping[str, Restorer],
    experience: Experience = "",
    settings: Settings | None = None,
    policy: Policy = None,
) -> tuple[ShotPlan, Image]:
    """Plan SAIR decisions for one image. Returns the plan and the original or restored image.

    Super-resolution is only decided here (`sr_factor`); applying it is the caller's job.
    A node the `policy` decides (strong measured evidence) makes no VLM call; every other node,
    and every node when the policy is None, behaves as without a policy.
    """
    target = (settings or load_settings()).thresholds.agent.sr_target_long_side
    decisions: list[str] = []

    profile = perception(vlm, trace, image)
    decisions.append(f"perception: {profile.key}")
    experience_text = experience(profile) if callable(experience) else experience

    active = policy(profile) if callable(policy) else policy
    chosen = image
    use_restored = False

    def decided(node: Literal["restorer", "sr"]) -> str | None:
        found = active.decision(node) if active is not None else None
        if found is None:
            return None
        record_decision(trace, found)
        decisions.append(found.rationale)
        return found.option

    restorer_verdict = decided("restorer")
    if restorer_verdict is None:
        restorer_name = restorer_select(vlm, trace, profile, experience_text).restorer
    else:
        restorer_name = active.mapped_restorer if active and restorer_verdict == MAPPED else NONE

    if restorer_name == NONE:
        decisions.append("restorer_select: none")
    elif restorer_name not in restorers:
        decisions.append(f"restorer_select: {restorer_name} unavailable, using none")
        restorer_name = NONE
    else:
        decisions.append(f"restorer_select: {restorer_name}")
        try:
            restored = restorers[restorer_name].restore(image)
        except Exception:
            logger.exception("restorer %s failed", restorer_name)
            decisions.append(f"restore: {restorer_name} failed, using none")
            restorer_name = NONE
        else:
            if restorer_verdict is not None:
                choice = "restored"  # the policy accepted the restored image without image_select
            else:
                choice = image_select(vlm, trace, profile, image, restored, experience_text).choice
            if choice == "restored":
                chosen, use_restored = restored, True
            decisions.append(f"image_select: {'restored' if use_restored else 'original'}")

    sr_verdict = decided("sr")
    if sr_verdict is not None:
        height, width = chosen.shape[:2]
        needed = required_factor(max(height, width), target) if sr_verdict == SR_AUTO else None
        factor: int | Literal["off"] = needed or "off"
    else:
        factor = sr_select(vlm, trace, chosen, profile, target, experience_text).factor
    decisions.append(f"sr_select: {factor}")

    plan = ShotPlan(
        restorer=restorer_name,
        use_restored=use_restored,
        sr_factor=None if factor == "off" else factor,
        decisions=decisions,
    )
    return plan, chosen
