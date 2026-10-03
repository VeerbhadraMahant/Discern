"""Role interfaces. Adapters wrap concrete models behind these; agent code only sees roles.

Images are HxWx3 uint8 numpy arrays. Boxes returned by detectors are absolute pixel xyxy in
the coordinate space of the image passed in; adapters convert at their boundary.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from discern.vision.boxes import Box

Image = npt.NDArray[np.uint8]
Embeddings = npt.NDArray[np.float32]


class Detection(BaseModel):
    model_config = ConfigDict(frozen=True)

    box: Box
    label: str
    score: float
    detector: str


class VLM(Protocol):
    def generate(self, prompt: str, images: Sequence[Image] = ()) -> str: ...


class Detector(Protocol):
    name: str

    def detect(self, image: Image, targets: Sequence[str]) -> list[Detection]: ...


class Restorer(Protocol):
    name: str

    def restore(self, image: Image) -> Image: ...


class SuperResolver(Protocol):
    def upscale(self, image: Image, factor: int) -> Image: ...


class Embedder(Protocol):
    def embed_images(self, images: Sequence[Image]) -> Embeddings: ...

    def embed_text(self, texts: Sequence[str]) -> Embeddings: ...
