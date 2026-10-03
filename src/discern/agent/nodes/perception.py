"""perception node: scene profile from a keyframe. Fallback: image statistics."""

from discern.agent.llm_io import load_prompt, structured_call
from discern.agent.schemas import SceneProfile
from discern.models.roles import VLM, Image
from discern.trace import TraceCollector
from discern.vision.stats import profile_from_stats


def perception(vlm: VLM, trace: TraceCollector, image: Image) -> SceneProfile:
    return structured_call(
        vlm,
        load_prompt("perception"),
        {},
        SceneProfile,
        lambda: profile_from_stats(image),
        trace,
        images=[image],
    )
