from google import genai
from google.genai import types
from pydantic import BaseModel

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.pipeline.retry import call_with_retry
from app.schemas import SceneContext, ToolPlan

_client: genai.Client | None = None

RESTORATION_CATALOG = {
    "low_light_enhancement": "Brightens and locally contrasts dark frames (CLAHE + gamma correction). Use for low_light/night lighting.",
    "dehaze": "Removes fog/dust haze via dark-channel prior. Use for fog/dust weather with reduced/poor visibility.",
    "denoise": "Reduces sensor grain/rain speckle noise. Use for rain or heavily noisy poor-visibility frames.",
}

DETECTION_CATALOG = {
    "person_detection": "Locates people AND vehicles (car, truck, bus, motorcycle, bicycle) in the frame (YOLOv8n). Use whenever human_presence, machinery_presence, or any street/traffic activity is true or uncertain.",
    "ppe_reasoning": "Vision-based check of each detected person for hard hat / high-vis vest compliance. Use whenever people are present near machinery or in a construction/industrial setting. Not applicable to vehicles.",
    "zone_reasoning": "Vision-based judgment of whether any person is inside a restricted/danger zone (near machinery, edges, excavation, traffic lane). Use when machinery_presence is true or the setting implies hazard zones (mining, ports, construction, roadways).",
}


class SceneAndPlan(BaseModel):
    scene: SceneContext
    plan: ToolPlan


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to backend/.env")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def classify_and_plan(image_bytes: bytes, media_type: str) -> tuple[SceneContext, ToolPlan]:
    client = _get_client()

    catalog_desc = "Restoration tools:\n" + "\n".join(f"- {k}: {v}" for k, v in RESTORATION_CATALOG.items())
    catalog_desc += "\n\nDetection tools:\n" + "\n".join(f"- {k}: {v}" for k, v in DETECTION_CATALOG.items())

    prompt = (
        "You are the perception-planning agent for Discern, a safety monitoring system for "
        "high-risk sites (construction, mining, ports, industrial plants). Classify this CCTV "
        "frame's environment and conditions, then select which processing tools to run from the "
        "catalog below. Skip restoration tools that would not help; only pick detection tools "
        "relevant to what's actually in the frame. restoration_tools and detection_tools must "
        "only contain values from the catalog keys.\n\n" + catalog_desc
    )

    response = call_with_retry(
        client.models.generate_content,
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=media_type),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SceneAndPlan,
        ),
    )

    result: SceneAndPlan = response.parsed or SceneAndPlan.model_validate_json(response.text)
    return result.scene, result.plan
