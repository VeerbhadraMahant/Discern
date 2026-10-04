"""MED for images: detector_select -> detect -> group -> adjudicate (system-design 5.2)."""

from collections.abc import Mapping, Sequence

from discern.agent.nodes.adjudicate import adjudicate as adjudicate_group
from discern.agent.nodes.detector_select import detector_select
from discern.agent.schemas import DetectorInfo, SceneProfile
from discern.config.settings import Settings, load_settings
from discern.models.roles import SCORE_FLOOR, VLM, Detection, Detector, Image
from discern.trace import TraceCollector
from discern.vision.grouping import group_detections


def detect_image(
    vlm: VLM,
    trace: TraceCollector,
    image: Image,
    targets: Sequence[str],
    profile: SceneProfile,
    detectors: Mapping[str, Detector],
    catalog: Sequence[DetectorInfo],
    adjudicate: bool = True,
    adjudicate_all: bool = True,
    settings: Settings | None = None,
    operating_thresholds: Mapping[str, float] | None = None,
    priority: Sequence[str] = (),
    experience: str = "",
    preferred: Sequence[str] | None = None,
) -> list[Detection]:
    """Multi-expertise detection. With adjudicate False, each group fuses to its anchor.

    With adjudicate_all False (a cost control, a deviation from the paper), groups where at
    least two detectors agree on the anchor's label are accepted without a VLM call.
    `preferred` is a detector set decided by the experience policy; it replaces detector_select.
    """
    settings = settings or load_settings()
    thresholds = operating_thresholds or {}
    default_floor = settings.thresholds.agent.default_operating_threshold
    available = [d for d in catalog if d.name in detectors]
    choice = detector_select(
        vlm, trace, targets, profile, available, priority, experience, settings, preferred
    )

    pooled: list[Detection] = []
    with trace.span("detect") as span:
        span.input_summary = f"detectors={choice.detectors}, targets={list(targets)}"
        for name in choice.detectors:
            detector = detectors.get(name)
            if detector is None:
                span.rationale += f"missing detector {name}; "
                continue
            floor = max(thresholds.get(name, default_floor), SCORE_FLOOR)
            pooled.extend(d for d in detector.detect(image, targets) if d.score >= floor)
        span.decision = f"{len(pooled)} detections above operating thresholds"

    with trace.span("group") as span:
        groups = group_detections(image, pooled, settings.thresholds.grouping)
        span.input_summary = f"detections={len(pooled)}"
        span.decision = f"{len(groups)} groups"

    if not adjudicate:
        return [g.anchor for g in groups]

    final: list[Detection] = []
    for group in groups:
        if not adjudicate_all and _detectors_agree(group.members, group.anchor.label):
            final.append(group.anchor)
            continue
        result = adjudicate_group(vlm, trace, image, group, targets, settings)
        if result is not None:
            final.append(
                Detection(
                    box=result.box,
                    label=result.label,
                    score=result.score,
                    detector=result.source_detector,
                )
            )
    return final


def _detectors_agree(members: Sequence[Detection], label: str) -> bool:
    return len({m.detector for m in members if m.label == label}) >= 2
