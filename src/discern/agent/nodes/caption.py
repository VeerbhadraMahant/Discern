"""caption node: one short VLM caption per shot, from its keyframe (system-design 5.4)."""

from discern.agent.llm_io import load_prompt, structured_call
from discern.agent.schemas import ShotCaption
from discern.models.roles import VLM, Image
from discern.trace import TraceCollector


def caption(vlm: VLM, trace: TraceCollector, keyframe: Image) -> str:
    """Caption a keyframe. Fallback: an empty caption, which later steps skip rather than invent."""
    result = structured_call(
        vlm,
        load_prompt("caption"),
        {},
        ShotCaption,
        lambda: ShotCaption.model_construct(caption=""),
        trace,
        images=[keyframe],
    )
    return result.caption.strip()
