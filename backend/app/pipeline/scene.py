import base64
import json

from anthropic import Anthropic

from app.config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from app.schemas import SceneContext, ToolPlan

_client: Anthropic | None = None

RESTORATION_CATALOG = {
    "low_light_enhancement": "Brightens and locally contrasts dark frames (CLAHE + gamma correction). Use for low_light/night lighting.",
    "dehaze": "Removes fog/dust haze via dark-channel prior. Use for fog/dust weather with reduced/poor visibility.",
    "denoise": "Reduces sensor grain/rain speckle noise. Use for rain or heavily noisy poor-visibility frames.",
}

DETECTION_CATALOG = {
    "person_detection": "Locates people in the frame (YOLOv8). Use whenever human_presence is true or uncertain.",
    "ppe_reasoning": "Vision-based check of each detected person for hard hat / high-vis vest compliance. Use whenever people are present near machinery or in a construction/industrial setting.",
    "zone_reasoning": "Vision-based judgment of whether any person is inside a restricted/danger zone (near machinery, edges, excavation). Use when machinery_presence is true or the setting implies hazard zones (mining, ports, construction).",
}

SCENE_TOOL = {
    "name": "report_scene_and_plan",
    "description": "Report the classified scene context and the selected processing plan for this CCTV frame.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scene": {
                "type": "object",
                "properties": {
                    "setting": {"type": "string", "description": "e.g. construction site, mining pit, port yard, industrial plant, unknown"},
                    "lighting": {"type": "string", "enum": ["bright", "normal", "low_light", "night"]},
                    "weather": {"type": "string", "enum": ["clear", "fog", "rain", "dust", "glare", "unknown"]},
                    "visibility": {"type": "string", "enum": ["good", "reduced", "poor"]},
                    "human_presence": {"type": "boolean"},
                    "machinery_presence": {"type": "boolean"},
                    "summary": {"type": "string", "description": "One sentence describing the scene."},
                },
                "required": ["setting", "lighting", "weather", "visibility", "human_presence", "machinery_presence", "summary"],
            },
            "plan": {
                "type": "object",
                "properties": {
                    "restoration_tools": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(RESTORATION_CATALOG.keys())},
                        "description": "Only include tools that would materially help. Empty array if the frame is already clear.",
                    },
                    "detection_tools": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(DETECTION_CATALOG.keys())},
                    },
                    "reasoning": {"type": "string", "description": "Brief justification for the selected tools, referencing the scene conditions."},
                },
                "required": ["restoration_tools", "detection_tools", "reasoning"],
            },
        },
        "required": ["scene", "plan"],
    },
}


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to backend/.env")
        _client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def classify_and_plan(image_bytes: bytes, media_type: str) -> tuple[SceneContext, ToolPlan]:
    client = _get_client()
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    catalog_desc = "Restoration tools:\n" + "\n".join(f"- {k}: {v}" for k, v in RESTORATION_CATALOG.items())
    catalog_desc += "\n\nDetection tools:\n" + "\n".join(f"- {k}: {v}" for k, v in DETECTION_CATALOG.items())

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        tools=[SCENE_TOOL],
        tool_choice={"type": "tool", "name": "report_scene_and_plan"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are the perception-planning agent for Discern, a safety monitoring system for "
                            "high-risk sites (construction, mining, ports, industrial plants). Classify this CCTV "
                            "frame's environment and conditions, then select which processing tools to run from the "
                            "catalog below. Skip restoration tools that would not help; only pick detection tools "
                            "relevant to what's actually in the frame.\n\n" + catalog_desc
                        ),
                    },
                ],
            }
        ],
    )

    tool_use = next(b for b in message.content if b.type == "tool_use")
    data = tool_use.input
    scene = SceneContext(**data["scene"])
    plan = ToolPlan(**data["plan"])
    return scene, plan
