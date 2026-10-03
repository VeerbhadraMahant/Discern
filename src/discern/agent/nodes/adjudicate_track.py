"""adjudicate_track node: one VLM call per track over its best crops, accept with a label or
reject (system-design 5.2 item 8)."""

from collections.abc import Sequence

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from discern.agent.llm_io import load_prompt, structured_call
from discern.agent.schemas import TrackAdjudication
from discern.config.settings import Settings, load_settings
from discern.models.roles import VLM, Image
from discern.trace import TraceCollector
from discern.video.tracks import MAX_CROPS
from discern.video.types import Track


def _numbered(crop: Image, number: int) -> Image:
    pil = PILImage.fromarray(crop).copy()
    ImageDraw.Draw(pil).text((3, 3), str(number), fill="red")
    return np.asarray(pil, dtype=np.uint8)


def _valid(result: TrackAdjudication, targets: Sequence[str]) -> bool:
    if not result.accept:
        return result.label is None
    return result.label is not None and result.label in targets


def adjudicate_track(
    vlm: VLM,
    trace: TraceCollector,
    track: Track,
    targets: Sequence[str],
    settings: Settings | None = None,
) -> TrackAdjudication:
    """Judge a track from up to three best crops (ranked by box area x score x sharpness).

    Fallback: accept with the majority detector label if the mean score reaches
    `fallback_accept_score`, else reject.
    """
    settings = settings or load_settings()
    accept = track.mean_score >= settings.thresholds.agent.fallback_accept_score
    fallback = TrackAdjudication(
        accept=accept,
        label=track.label if accept else None,
        rationale="fallback: mean detector score threshold",
    )
    crops = track.best_crops[:MAX_CROPS]
    crop_lines = "\n".join(
        f"{i}. t={c.time:.2f}s, box area x score x sharpness rank={c.rank:.0f}"
        for i, c in enumerate(crops, start=1)
    )
    summary = (
        f"{len(track.frames)} sampled frames from {track.t_start:.2f}s to {track.t_end:.2f}s, "
        f"detector label {track.label}, mean score {track.mean_score:.2f}"
    )
    result = structured_call(
        vlm,
        load_prompt("adjudicate_track"),
        {"targets": ", ".join(targets), "track": summary, "crops": crop_lines},
        TrackAdjudication,
        lambda: fallback,
        trace,
        images=[_numbered(c.image, i) for i, c in enumerate(crops, start=1)],
    )
    if not _valid(result, targets):
        with trace.span("adjudicate_track.guard") as span:
            span.input_summary = f"vlm_choice={result.model_dump_json()}"
            span.fallback_used = True
            span.decision = fallback.model_dump_json()
            span.rationale = "VLM answer failed code validation"
        result = fallback
    return result
