"""adjudicate node: the VLM picks one candidate box per instance group, or rejects the group."""

from collections.abc import Sequence

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
from pydantic import BaseModel, ConfigDict

from discern.agent.llm_io import load_prompt, structured_call
from discern.agent.schemas import AdjudicationChoice
from discern.config.settings import Settings, load_settings
from discern.models.roles import VLM, Image
from discern.trace import TraceCollector
from discern.vision.boxes import Box, clip, expand, union_box
from discern.vision.grouping import InstanceGroup

# Distinct colours for numbered candidate boxes (cycled).
_COLOURS = ("red", "lime", "blue", "yellow", "magenta", "cyan")


class AdjudicatedDetection(BaseModel):
    model_config = ConfigDict(frozen=True)

    box: Box
    label: str
    source_detector: str
    score: float
    rejected: bool = False


def _context_region(region: Box, min_px: int, width: int, height: int) -> Box:
    """Widen a region around its centre to at least min_px per side (kept inside the image)."""
    cx, cy = (region.x1 + region.x2) / 2, (region.y1 + region.y2) / 2
    half_w, half_h = max(region.width, min_px) / 2, max(region.height, min_px) / 2
    return clip(Box(cx - half_w, cy - half_h, cx + half_w, cy + half_h), width, height)


def _crop_with_candidates(
    image: Image, group: InstanceGroup, alpha: float, min_context_px: int, min_view_px: int
) -> Image:
    height, width = image.shape[:2]
    union = expand(union_box([m.box for m in group.members]), alpha)
    region = _context_region(clip(union, width, height), min_context_px, width, height)
    x1, y1 = min(int(region.x1), width - 1), min(int(region.y1), height - 1)
    x2, y2 = (
        min(max(int(region.x2) + 1, x1 + 1), width),
        min(max(int(region.y2) + 1, y1 + 1), height),
    )
    pil = PILImage.fromarray(image[y1:y2, x1:x2]).copy()
    scale = max(1.0, min_view_px / max(1, min(pil.width, pil.height)))
    if scale > 1.0:
        pil = pil.resize(
            (round(pil.width * scale), round(pil.height * scale)), PILImage.Resampling.LANCZOS
        )
    draw = ImageDraw.Draw(pil)
    for i, m in enumerate(group.members):
        colour = _COLOURS[i % len(_COLOURS)]
        b = tuple(v * scale for v in (m.box.x1 - x1, m.box.y1 - y1, m.box.x2 - x1, m.box.y2 - y1))
        draw.rectangle(b, outline=colour, width=2)
        draw.text((b[0] + 3, b[1] + 3), str(i + 1), fill=colour)
    return np.asarray(pil, dtype=np.uint8)


def _valid(choice: AdjudicationChoice, n: int, targets: Sequence[str]) -> bool:
    if choice.reject:
        return choice.candidate is None
    return (
        choice.candidate is not None
        and 1 <= choice.candidate <= n
        and choice.label is not None
        and choice.label in targets
    )


def adjudicate(
    vlm: VLM,
    trace: TraceCollector,
    image: Image,
    group: InstanceGroup,
    targets: Sequence[str],
    settings: Settings | None = None,
) -> AdjudicatedDetection | None:
    """Return the chosen detection, or None when the group is rejected."""
    settings = settings or load_settings()
    anchor = group.anchor
    accept_anchor = anchor.score >= settings.thresholds.agent.fallback_accept_score
    fallback = AdjudicationChoice(
        candidate=1 if accept_anchor else None,
        label=anchor.label if accept_anchor else None,
        reject=not accept_anchor,
        rationale="fallback: anchor score threshold",
    )
    candidates = "\n".join(
        f"{i}. detector={m.detector}, label={m.label}, score={m.score:.2f}"
        for i, m in enumerate(group.members, start=1)
    )
    agent_cfg = settings.thresholds.agent
    crop = _crop_with_candidates(
        image,
        group,
        settings.thresholds.grouping.alpha,
        agent_cfg.adjudicate_min_context_px,
        agent_cfg.adjudicate_min_view_px,
    )
    choice = structured_call(
        vlm,
        load_prompt("adjudicate"),
        {"targets": ", ".join(targets), "candidates": candidates},
        AdjudicationChoice,
        lambda: fallback,
        trace,
        images=[crop],
    )
    if not _valid(choice, len(group.members), targets):
        with trace.span("adjudicate.guard") as span:
            span.input_summary = f"vlm_choice={choice.model_dump_json()}"
            span.fallback_used = True
            span.decision = fallback.model_dump_json()
            span.rationale = "VLM answer failed code validation"
        choice = fallback
    if choice.reject or choice.candidate is None or choice.label is None:
        return None
    member = group.members[choice.candidate - 1]
    # The VLM picks among the candidates; it may not invent a label none of them proposed.
    candidate_labels = {m.label for m in group.members}
    label = choice.label if choice.label in candidate_labels else member.label
    return AdjudicatedDetection(
        box=member.box,
        label=label,
        source_detector=member.detector,
        score=member.score,
    )
