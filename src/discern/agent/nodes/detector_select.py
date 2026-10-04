"""detector_select node: the VLM picks top-K detectors; code enforces the catalog and K.

Without experience the VLM has no evidence to choose by (measured: its unaided picks were worse
than the deterministic priority order), so the priority order is used and no VLM call is made.
A caller may pass `preferred` (a set decided by the experience policy); it is used as is, with no
VLM call, when it names exactly K catalog detectors.
"""

from collections.abc import Sequence

from discern.agent.llm_io import load_prompt, structured_call
from discern.agent.schemas import DetectorChoice, DetectorInfo, SceneProfile
from discern.config.settings import Settings, load_settings
from discern.models.roles import VLM
from discern.trace import TraceCollector


def _ranking(catalog: Sequence[DetectorInfo], priority: Sequence[str]) -> list[str]:
    """Catalog names ordered by `priority`, then any remaining names in catalog order."""
    names = [d.name for d in catalog]
    ranked = [n for n in dict.fromkeys(priority) if n in names]
    return ranked + [n for n in names if n not in ranked]


def detector_select(
    vlm: VLM,
    trace: TraceCollector,
    targets: Sequence[str],
    profile: SceneProfile,
    catalog: Sequence[DetectorInfo],
    priority: Sequence[str],
    experience: str = "",
    settings: Settings | None = None,
    preferred: Sequence[str] | None = None,
) -> DetectorChoice:
    k = (settings or load_settings()).thresholds.agent.top_k_detectors
    ranking = _ranking(catalog, priority)
    fallback = DetectorChoice(detectors=ranking[:k], rationale="fixed priority order")
    if preferred is not None and len(set(preferred)) == k and set(preferred) <= set(ranking):
        # An experience policy decided the set (strong measured evidence): no VLM call.
        chosen = DetectorChoice(
            detectors=list(preferred), rationale="experience policy: measured best detector set"
        )
        with trace.span("detector_select") as span:
            span.input_summary = f"targets={list(targets)}, profile={profile.key}"
            span.decision = chosen.model_dump_json()
            span.rationale = chosen.rationale
        return chosen
    if not experience.strip():
        with trace.span("detector_select") as span:
            span.input_summary = f"targets={list(targets)}, profile={profile.key}"
            span.decision = fallback.model_dump_json()
            span.rationale = "no experience available: deterministic priority order, no VLM call"
        return fallback
    catalog_text = "\n".join(f"- {d.name} ({d.speed_class}): {d.capabilities}" for d in catalog)

    choice = structured_call(
        vlm,
        load_prompt("detector_select"),
        {
            "targets": ", ".join(targets),
            "profile": profile.model_dump_json(),
            "catalog": catalog_text,
            "k": k,
            "experience": experience,
        },
        DetectorChoice,
        lambda: fallback,
        trace,
    )
    valid = [n for n in dict.fromkeys(choice.detectors) if n in ranking]
    final = (valid + [n for n in ranking if n not in valid])[:k]
    if final == choice.detectors:
        return choice
    with trace.span("detector_select.guard") as span:
        span.input_summary = f"vlm_choice={choice.detectors}"
        span.fallback_used = True
        span.decision = str(final)
        span.rationale = "dropped unknown or duplicate names, padded from priority order to K"
    return DetectorChoice(detectors=final, rationale=choice.rationale)
