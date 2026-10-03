"""Deterministic fakes for CPU tests of agent logic."""

import zlib
from collections.abc import Sequence

import numpy as np

from discern.models.roles import Detection, Embeddings, Image


class FakeVLM:
    """Returns scripted responses in order; records every prompt it receives."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, images: Sequence[Image] = ()) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("FakeVLM ran out of scripted responses")
        return self._responses.pop(0)


class FakeDetector:
    def __init__(self, name: str, detections: Sequence[Detection] = ()) -> None:
        self.name = name
        self._detections = list(detections)

    def detect(self, image: Image, targets: Sequence[str]) -> list[Detection]:
        return [d for d in self._detections if d.label in targets]


class FakeRestorer:
    """Brightens by a fixed offset so tests can tell restored from original."""

    def __init__(self, name: str = "fake_restorer", offset: int = 10) -> None:
        self.name = name
        self._offset = offset

    def restore(self, image: Image) -> Image:
        return np.clip(image.astype(np.int16) + self._offset, 0, 255).astype(np.uint8)


class FakeSuperResolver:
    def upscale(self, image: Image, factor: int) -> Image:
        return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


class FakeEmbedder:
    """Hash-seeded unit vectors: same input gives the same embedding."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def _unit(self, seed: int) -> Embeddings:
        v = np.random.default_rng(seed).normal(size=self._dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_images(self, images: Sequence[Image]) -> Embeddings:
        return np.stack([self._unit(int(img.sum()) % (2**32)) for img in images])

    def embed_text(self, texts: Sequence[str]) -> Embeddings:
        return np.stack([self._unit(zlib.crc32(t.encode())) for t in texts])
