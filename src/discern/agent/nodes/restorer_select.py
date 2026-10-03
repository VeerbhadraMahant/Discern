"""restorer_select node: rule map scene label -> restorer, VLM may only veto it."""

from discern.agent.llm_io import load_prompt, structured_call
from discern.agent.schemas import RestorerChoice, SceneLabel, SceneProfile
from discern.models.roles import VLM
from discern.trace import TraceCollector

NONE = "none"
RESTORER_FOR_SCENE: dict[SceneLabel, str] = {
    "fog": "dehaze",
    "rain": "derain",
    "noise": "denoise",
    "low_light": "lowlight",
    "normal": NONE,
    "underwater": NONE,
}


def restorer_select(
    vlm: VLM, trace: TraceCollector, profile: SceneProfile, experience: str = ""
) -> RestorerChoice:
    default = RESTORER_FOR_SCENE[profile.scene_label]
    fallback = RestorerChoice(restorer=default, rationale="rule map")
    if default == NONE:
        with trace.span("restorer_select") as span:
            span.input_summary = f"profile={profile.key}"
            span.decision = fallback.model_dump_json()
            span.rationale = "no restorer mapped for this scene label"
        return fallback

    choice = structured_call(
        vlm,
        load_prompt("restorer_select"),
        {"profile": profile.model_dump_json(), "default": default, "experience": experience},
        RestorerChoice,
        lambda: fallback,
        trace,
    )
    if choice.restorer in (NONE, default):
        return choice
    with trace.span("restorer_select.guard") as span:
        span.input_summary = f"vlm_choice={choice.restorer}"
        span.fallback_used = True
        span.decision = fallback.model_dump_json()
        span.rationale = "VLM named a restorer outside the allowed set"
    return fallback
