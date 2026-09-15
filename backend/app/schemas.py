from typing import Literal
from pydantic import BaseModel


class SceneContext(BaseModel):
    setting: str
    lighting: Literal["bright", "normal", "low_light", "night"]
    weather: Literal["clear", "fog", "rain", "dust", "glare", "unknown"]
    visibility: Literal["good", "reduced", "poor"]
    human_presence: bool
    machinery_presence: bool
    summary: str


class ToolPlan(BaseModel):
    restoration_tools: list[Literal["low_light_enhancement", "dehaze", "denoise"]]
    detection_tools: list[Literal["person_detection", "ppe_reasoning", "zone_reasoning"]]
    reasoning: str


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Detection(BaseModel):
    id: str
    label: str
    confidence: float
    box: BoundingBox
    source: str  # which tool produced this detection


class Violation(BaseModel):
    id: str
    type: Literal["missing_ppe", "restricted_zone", "unsafe_proximity"]
    severity: Literal["critical", "warning", "info"]
    description: str
    related_detection_ids: list[str]


class PipelineStep(BaseModel):
    name: str
    status: Literal["skipped", "completed"]
    detail: str
    duration_ms: int


class AnalyzeResponse(BaseModel):
    scene: SceneContext
    plan: ToolPlan
    steps: list[PipelineStep]
    detections: list[Detection]
    violations: list[Violation]
    annotated_image: str  # base64 data URL
