"""SAIR orchestration: perception, restorer and image selection, SR selection."""

import logging
from collections.abc import Callable, Mapping

from discern.agent.nodes.image_select import image_select
from discern.agent.nodes.perception import perception
from discern.agent.nodes.restorer_select import NONE, restorer_select
from discern.agent.nodes.sr_select import sr_select
from discern.agent.schemas import SceneProfile, ShotPlan
from discern.config.settings import Settings, load_settings
from discern.models.roles import VLM, Image, Restorer
from discern.trace import TraceCollector

logger = logging.getLogger(__name__)

# Experience text for the restorer and SR prompts: fixed, or computed from the perceived profile.
Experience = str | Callable[[SceneProfile], str]


def plan_image(
    vlm: VLM,
    trace: TraceCollector,
    image: Image,
    restorers: Mapping[str, Restorer],
    experience: Experience = "",
    settings: Settings | None = None,
) -> tuple[ShotPlan, Image]:
    """Plan SAIR decisions for one image. Returns the plan and the original or restored image.

    Super-resolution is only decided here (`sr_factor`); applying it is the caller's job.
    """
    target = (settings or load_settings()).thresholds.agent.sr_target_long_side
    decisions: list[str] = []

    profile = perception(vlm, trace, image)
    decisions.append(f"perception: {profile.key}")
    experience_text = experience(profile) if callable(experience) else experience

    restorer_name = restorer_select(vlm, trace, profile, experience_text).restorer
    chosen = image
    use_restored = False

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
            choice = image_select(vlm, trace, profile, image, restored, experience_text).choice
            if choice == "restored":
                chosen, use_restored = restored, True
            decisions.append(f"image_select: {'restored' if use_restored else 'original'}")

    sr = sr_select(vlm, trace, chosen, profile, target, experience_text)
    decisions.append(f"sr_select: {sr.factor}")

    plan = ShotPlan(
        restorer=restorer_name,
        use_restored=use_restored,
        sr_factor=None if sr.factor == "off" else sr.factor,
        decisions=decisions,
    )
    return plan, chosen
