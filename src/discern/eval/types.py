"""Common annotation format shared by every dataset loader."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from discern.vision.boxes import Box


class GroundTruthBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    box: Box
    label: str


class AnnotatedImage(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_id: str
    path: Path
    width: int
    height: int
    objects: tuple[GroundTruthBox, ...]
    scene_label: str | None = None  # dataset-implied condition: fog, low_light, rain, normal, ...
