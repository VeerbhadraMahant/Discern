import base64
import time

import cv2
import numpy as np

from app.pipeline import detection, reasoning, scene as scene_module
from app.pipeline.restoration import RESTORATION_FUNCS
from app.schemas import AnalyzeResponse, PipelineStep

SEVERITY_COLOR = {
    "critical": (33, 33, 220),   # BGR red
    "warning": (0, 165, 255),    # BGR amber
    "info": (150, 150, 150),
}
OK_COLOR = (110, 190, 90)  # BGR green


def run_pipeline(image_bytes: bytes, media_type: str) -> AnalyzeResponse:
    steps: list[PipelineStep] = []

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    t0 = time.perf_counter()
    scene, plan = scene_module.classify_and_plan(image_bytes, media_type)
    steps.append(
        PipelineStep(
            name="Scene understanding",
            status="completed",
            detail=scene.summary,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
    )
    steps.append(
        PipelineStep(
            name="Tool selection",
            status="completed",
            detail=plan.reasoning,
            duration_ms=0,
        )
    )

    working_img = img
    if plan.restoration_tools:
        t0 = time.perf_counter()
        applied = []
        for tool_name in plan.restoration_tools:
            fn = RESTORATION_FUNCS.get(tool_name)
            if fn:
                working_img = fn(working_img)
                applied.append(tool_name)
        steps.append(
            PipelineStep(
                name="Restoration",
                status="completed",
                detail=f"Applied: {', '.join(applied)}" if applied else "No-op",
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
        )
    else:
        steps.append(PipelineStep(name="Restoration", status="skipped", detail="Frame conditions did not warrant restoration", duration_ms=0))

    detections = []
    if "person_detection" in plan.detection_tools:
        t0 = time.perf_counter()
        detections = detection.person_detection(working_img)
        n_people = sum(1 for d in detections if d.label == "person")
        n_vehicles = len(detections) - n_people
        steps.append(
            PipelineStep(
                name="Object detection",
                status="completed",
                detail=f"{n_people} person(s), {n_vehicles} vehicle(s) detected",
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
        )
    else:
        steps.append(PipelineStep(name="Object detection", status="skipped", detail="No human/vehicle presence expected", duration_ms=0))

    check_ppe = "ppe_reasoning" in plan.detection_tools
    check_zones = "zone_reasoning" in plan.detection_tools
    persons = detection.people_only(detections)
    violations = []
    if check_ppe or check_zones:
        t0 = time.perf_counter()
        ok, encoded = cv2.imencode(".jpg", working_img)
        restored_bytes = encoded.tobytes() if ok else image_bytes
        violations = reasoning.assess_violations(restored_bytes, "image/jpeg", scene, persons, check_ppe, check_zones)
        steps.append(
            PipelineStep(
                name="Violation reasoning",
                status="completed",
                detail=f"{len(violations)} violation(s) found" if persons else "No people to assess",
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
        )
    else:
        steps.append(PipelineStep(name="Violation reasoning", status="skipped", detail="No PPE/zone checks selected", duration_ms=0))

    annotated = _annotate(working_img, detections, violations)
    ok, encoded = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.standard_b64encode(encoded.tobytes()).decode("utf-8") if ok else ""

    return AnalyzeResponse(
        scene=scene,
        plan=plan,
        steps=steps,
        detections=detections,
        violations=violations,
        annotated_image=f"data:image/jpeg;base64,{annotated_b64}",
    )


VEHICLE_COLOR = (200, 140, 0)  # BGR cyan-ish blue, distinct from person severity colors


def _annotate(img: np.ndarray, detections: list, violations: list) -> np.ndarray:
    out = img.copy()
    flagged_ids = {vid for v in violations for vid in v.related_detection_ids}
    violation_by_detection: dict[str, str] = {}
    for v in violations:
        for did in v.related_detection_ids:
            violation_by_detection[did] = v.severity

    for d in detections:
        x, y, w, h = int(d.box.x), int(d.box.y), int(d.box.width), int(d.box.height)
        if d.label == "person":
            severity = violation_by_detection.get(d.id)
            color = SEVERITY_COLOR.get(severity, OK_COLOR) if d.id in flagged_ids else OK_COLOR
        else:
            color = VEHICLE_COLOR
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{d.label} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, max(0, y - th - 8)), (x + tw + 6, y), color, -1)
        cv2.putText(out, label, (x + 3, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out
