"""sr_select node: code computes the factor to reach the target, the VLM only says on or off."""

import math

from discern.agent.llm_io import load_prompt, structured_call
from discern.agent.schemas import SceneProfile, SRChoice
from discern.models.roles import VLM, Image
from discern.models.tiling import SR_FACTORS
from discern.trace import TraceCollector


def required_factor(long_side: int, target_long_side: int) -> int | None:
    """Smallest supported integer factor reaching the target (clamped to 4); None if not needed."""
    if long_side >= target_long_side:
        return None
    needed = math.ceil(target_long_side / long_side)
    return SR_FACTORS[0] if needed <= SR_FACTORS[0] else SR_FACTORS[-1]


def sr_select(
    vlm: VLM,
    trace: TraceCollector,
    image: Image,
    profile: SceneProfile,
    target_long_side: int,
    experience: str = "",
) -> SRChoice:
    height, width = image.shape[:2]
    factor = required_factor(max(height, width), target_long_side)
    if factor is None:
        off = SRChoice(factor="off", rationale="image already reaches the target size")
        with trace.span("sr_select") as span:
            span.input_summary = f"size={width}x{height}, target={target_long_side}"
            span.decision = off.model_dump_json()
            span.rationale = off.rationale
        return off

    fallback = SRChoice(factor=factor, rationale="fallback: image below target size")
    # The v2 prompt makes measured experience the primary evidence; without experience v1 is used.
    choice = structured_call(
        vlm,
        load_prompt("sr_select", 2 if experience.strip() else 1),
        {
            "profile": profile.model_dump_json(),
            "width": width,
            "height": height,
            "target": target_long_side,
            "factor": factor,
            "experience": experience,
        },
        SRChoice,
        lambda: fallback,
        trace,
    )
    if choice.factor in ("off", factor):
        return choice
    with trace.span("sr_select.guard") as span:
        span.input_summary = f"vlm_factor={choice.factor}, computed={factor}"
        span.fallback_used = True
        span.decision = SRChoice(factor=factor, rationale=choice.rationale).model_dump_json()
        span.rationale = "VLM factor differs from the computed factor; using the computed one"
    return SRChoice(factor=factor, rationale=choice.rationale)
