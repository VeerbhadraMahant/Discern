"""Pydantic schemas for the SAIR nodes (system-design 5.2)."""

from typing import Literal

from pydantic import BaseModel, Field

SceneLabel = Literal["normal", "fog", "rain", "underwater", "low_light", "noise"]
Illumination = Literal["dark", "dim", "normal", "bright"]
Visibility = Literal["clear", "moderate", "poor"]
ObjectScale = Literal["small", "medium", "large", "mixed"]
ObjectDensity = Literal["sparse", "moderate", "dense"]


class SceneProfile(BaseModel):
    scene_label: SceneLabel
    illumination: Illumination
    visibility: Visibility
    object_scale: ObjectScale
    object_density: ObjectDensity
    confidence: float = Field(ge=0.0, le=1.0)

    @property
    def key(self) -> str:
        """Discrete profile key used for experience retrieval."""
        return "|".join(
            (
                self.scene_label,
                self.illumination,
                self.visibility,
                self.object_scale,
                self.object_density,
            )
        )


class RestorerChoice(BaseModel):
    restorer: str  # a restorer role name, or "none"
    rationale: str = ""


class ImageChoice(BaseModel):
    choice: Literal["original", "restored"]
    rationale: str = ""


class SRChoice(BaseModel):
    factor: int | Literal["off"]  # "off" or an integer upscale factor
    rationale: str = ""


class ShotPlan(BaseModel):
    restorer: str  # "none" when no restoration was applied
    use_restored: bool
    sr_factor: int | None  # None means super-resolution is off
    decisions: list[str] = Field(default_factory=list)


class DetectorInfo(BaseModel):
    """One catalog entry shown to the detector_select prompt."""

    name: str
    capabilities: str
    speed_class: Literal["fast", "slow"]


class DetectorChoice(BaseModel):
    detectors: list[str]
    rationale: str = ""


class AdjudicationChoice(BaseModel):
    candidate: int | None = None  # 1-based candidate number
    label: str | None = None
    reject: bool = False
    rationale: str = ""


class TrackAdjudication(BaseModel):
    accept: bool
    label: str | None = None
    rationale: str = ""


class ShotCaption(BaseModel):
    caption: str = Field(min_length=1)


class AttributeValue(BaseModel):
    value: str = Field(min_length=1)  # the attribute's value for one tracked object
    rationale: str = ""


class AnswerText(BaseModel):
    answer: str = Field(min_length=1)
