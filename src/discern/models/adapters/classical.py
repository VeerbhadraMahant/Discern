"""Classical restorers from legacy/v0 behind the Restorer protocol. No weights, CPU only."""

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from discern.models.manager import RegistryEntry
from discern.models.roles import Image

_LEGACY = Path(__file__).resolve().parents[4] / "legacy" / "v0"
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))

import restoration  # noqa: E402  (legacy/v0/restoration.py, importable after the path edit)


class _ClassicalRestorer:
    _func: Callable[[np.ndarray], np.ndarray]

    def __init__(self, entry: RegistryEntry, device: str = "cpu") -> None:
        self.name = entry.name

    def restore(self, image: Image) -> Image:
        # The legacy functions are written for OpenCV BGR arrays.
        out = type(self)._func(np.ascontiguousarray(image[:, :, ::-1]))
        return np.ascontiguousarray(out[:, :, ::-1])


class ClassicalDehazeAdapter(_ClassicalRestorer):
    _func = staticmethod(restoration.dehaze)


class ClassicalLowLightAdapter(_ClassicalRestorer):
    _func = staticmethod(restoration.low_light_enhancement)


class ClassicalDenoiseAdapter(_ClassicalRestorer):
    _func = staticmethod(restoration.denoise)
