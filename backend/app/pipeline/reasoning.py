import base64
import uuid

from anthropic import Anthropic

from app.config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from app.schemas import Detection, SceneContext, Violation

_client: Anthropic | None = None

VIOLATION_TOOL = {
    "name": "report_violations",
    "description": "Report safety violations found by inspecting the frame and the detected people.",
    "input_schema": {
        "type": "object",
        "properties": {
            "violations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["missing_ppe", "restricted_zone", "unsafe_proximity"]},
                        "severity": {"type": "string", "enum": ["critical", "warning", "info"]},
                        "description": {"type": "string"},
                        "related_detection_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["type", "severity", "description", "related_detection_ids"],
                },
            }
        },
        "required": ["violations"],
    },
}


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to backend/.env")
        _client = Anthropic(api_key=ANTHROPIC_API_KEY)
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
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    person_lines = "\n".join(
        f"- id={p.id}, box(x,y,w,h)=({p.box.x:.0f},{p.box.y:.0f},{p.box.width:.0f},{p.box.height:.0f})"
        for p in persons
    )
    checks = []
    if check_ppe:
        checks.append("- PPE compliance: is each person wearing a hard hat and high-visibility vest? Flag missing_ppe if not.")
    if check_zones:
        checks.append("- Restricted/danger zones: is any person inside a hazard zone (directly beside/under machinery, on an unguarded edge, inside an excavation)? Flag restricted_zone. If a person is close enough to moving/active machinery to be struck, flag unsafe_proximity instead.")

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        tools=[VIOLATION_TOOL],
        tool_choice={"type": "tool", "name": "report_violations"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {
                        "type": "text",
                        "text": (
                            f"Scene: {scene.summary} (setting={scene.setting}, lighting={scene.lighting}, "
                            f"weather={scene.weather}, visibility={scene.visibility}).\n\n"
                            f"Detected people (pixel coordinates, origin top-left):\n{person_lines}\n\n"
                            "Checks to perform:\n" + "\n".join(checks) + "\n\n"
                            "Only report violations you can actually see evidence for. Reference person ids in "
                            "related_detection_ids. If nothing is wrong, report an empty list."
                        ),
                    },
                ],
            }
        ],
    )

    tool_use = next(b for b in message.content if b.type == "tool_use")
    raw = tool_use.input["violations"]
    return [Violation(id=f"violation_{uuid.uuid4().hex[:8]}", **v) for v in raw]
