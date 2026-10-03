"""attribute_check node: classify one attribute of one tracked object from its best crops."""

from discern.agent.llm_io import load_prompt, structured_call
from discern.agent.schemas import AttributeValue
from discern.models.roles import VLM
from discern.trace import TraceCollector
from discern.video.tracks import MAX_CROPS
from discern.video.types import Track

UNKNOWN = "unknown"


def attribute_check(vlm: VLM, trace: TraceCollector, track: Track, attribute: str) -> str:
    """The attribute's value for the track as a lowercase string.

    Fallback: "unknown", which never matches a requested value, so an unclassifiable track is
    excluded from a filter rather than guessed into it.
    """
    result = structured_call(
        vlm,
        load_prompt("attribute_check"),
        {"label": track.label, "attribute": attribute},
        AttributeValue,
        lambda: AttributeValue(value=UNKNOWN, rationale="fallback: classification unavailable"),
        trace,
        images=[c.image for c in track.best_crops[:MAX_CROPS]],
    )
    return result.value.strip().lower()
