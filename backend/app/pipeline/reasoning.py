import uuid
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.schemas import Detection, SceneContext, Violation

_client: genai.Client | None = None


class ViolationItem(BaseModel):
    type: Literal["missing_ppe", "restricted_zone", "unsafe_proximity"]
    severity: Literal["critical", "warning", "info"]
    description: str
    related_detection_ids: list[str]


class ViolationsResponse(BaseModel):
    violations: list[ViolationItem]


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to backend/.env")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def assess_violations(
    image_bytes: bytes,
    media_type: str,
    scene: SceneContext,
    persons: list[Detection],
    check_ppe: bool,
    check_zones: bool,
) -> list[Violation]:
    if not persons or (not check_ppe and not check_zones):
        return []

    client = _get_client()

    person_lines = "\n".join(
        f"- id={p.id}, box(x,y,w,h)=({p.box.x:.0f},{p.box.y:.0f},{p.box.width:.0f},{p.box.height:.0f})"
        for p in persons
    )
    checks = []
    if check_ppe:
        checks.append("- PPE compliance: is each person wearing a hard hat and high-visibility vest? Flag missing_ppe if not.")
    if check_zones:
        checks.append("- Restricted/danger zones: is any person inside a hazard zone (directly beside/under machinery, on an unguarded edge, inside an excavation)? Flag restricted_zone. If a person is close enough to moving/active machinery to be struck, flag unsafe_proximity instead.")

    prompt = (
        f"Scene: {scene.summary} (setting={scene.setting}, lighting={scene.lighting}, "
        f"weather={scene.weather}, visibility={scene.visibility}).\n\n"
        f"Detected people (pixel coordinates, origin top-left):\n{person_lines}\n\n"
        "Checks to perform:\n" + "\n".join(checks) + "\n\n"
        "Only report violations you can actually see evidence for. Reference person ids in "
        "related_detection_ids. If nothing is wrong, return an empty violations list."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=media_type),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ViolationsResponse,
        ),
    )

    result: ViolationsResponse = response.parsed or ViolationsResponse.model_validate_json(response.text)
    return [Violation(id=f"violation_{uuid.uuid4().hex[:8]}", **v.model_dump()) for v in result.violations]
