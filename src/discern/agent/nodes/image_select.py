"""image_select node: original vs restored, judged on detection-oriented criteria."""

from discern.agent.llm_io import load_prompt, structured_call
from discern.agent.schemas import ImageChoice, SceneProfile
from discern.models.roles import VLM, Image
from discern.trace import TraceCollector


def image_select(
    vlm: VLM, trace: TraceCollector, profile: SceneProfile, original: Image, restored: Image
) -> ImageChoice:
    """Image A is the original, image B the restored one. Fallback: original."""
    return structured_call(
        vlm,
        load_prompt("image_select"),
        {"profile": profile.model_dump_json()},
        ImageChoice,
        lambda: ImageChoice(choice="original", rationale="fallback: restoration is opt-in"),
        trace,
        images=[original, restored],
    )
