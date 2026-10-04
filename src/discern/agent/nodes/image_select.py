"""image_select node: original vs restored, judged on detection-oriented criteria."""

from discern.agent.llm_io import load_prompt, structured_call
from discern.agent.schemas import ImageChoice, SceneProfile
from discern.models.roles import VLM, Image
from discern.trace import TraceCollector


def image_select(
    vlm: VLM,
    trace: TraceCollector,
    profile: SceneProfile,
    original: Image,
    restored: Image,
    experience: str = "",
) -> ImageChoice:
    """Image A is the original, image B the restored one. Fallback: original.

    With experience text the v2 prompt (which shows it) is used; without it the v1 prompt is
    used unchanged, so DetAS without experience behaves exactly as before.
    """
    with_experience = bool(experience.strip())
    variables: dict[str, object] = {"profile": profile.model_dump_json()}
    if with_experience:
        variables["experience"] = experience
    return structured_call(
        vlm,
        load_prompt("image_select", 2 if with_experience else 1),
        variables,
        ImageChoice,
        lambda: ImageChoice(choice="original", rationale="fallback: restoration is opt-in"),
        trace,
        images=[original, restored],
    )
