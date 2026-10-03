"""Pure-numpy image statistics and the heuristic scene profile used as perception fallback.

Images are HxWx3 uint8 RGB. Luminance, contrast and dark channel are reported in 0..1,
the noise estimate in 8-bit intensity units.
"""

import numpy as np
import numpy.typing as npt

from discern.agent.schemas import SceneProfile
from discern.models.roles import Image

DARK_CHANNEL_PATCH = 15

LOW_LIGHT_LUMINANCE = 0.25
NOISE_SIGMA = 10.0
FOG_DARK_CHANNEL = 0.4
FOG_CONTRAST = 0.15
FOG_MIN_LUMINANCE = 0.45
HEURISTIC_CONFIDENCE = 0.3

_FloatArray = npt.NDArray[np.float64]


def _luma(image: Image) -> _FloatArray:
    rgb = image.astype(np.float64)
    return np.asarray(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def mean_luminance(image: Image) -> float:
    return float(_luma(image).mean() / 255.0)


def contrast(image: Image) -> float:
    """Standard deviation of luminance."""
    return float(_luma(image).std() / 255.0)


def dark_channel_strength(image: Image, patch: int = DARK_CHANNEL_PATCH) -> float:
    """Mean of the per-patch minimum over channels and pixels; high values indicate haze."""
    minimum = image.min(axis=2)
    h, w = minimum.shape
    ps_h, ps_w = min(patch, h), min(patch, w)
    ph, pw = h // ps_h, w // ps_w
    blocks = minimum[: ph * ps_h, : pw * ps_w].reshape(ph, ps_h, pw, ps_w)
    return float(blocks.min(axis=(1, 3)).mean() / 255.0)


def noise_estimate(image: Image) -> float:
    """Immerkaer fast noise estimate, returned as a standard deviation."""
    y = _luma(image)
    h, w = y.shape
    if h < 3 or w < 3:
        return 0.0
    conv = (
        y[:-2, :-2]
        - 2 * y[:-2, 1:-1]
        + y[:-2, 2:]
        - 2 * y[1:-1, :-2]
        + 4 * y[1:-1, 1:-1]
        - 2 * y[1:-1, 2:]
        + y[2:, :-2]
        - 2 * y[2:, 1:-1]
        + y[2:, 2:]
    )
    return float(np.sqrt(np.pi / 2.0) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2)))


def profile_from_stats(image: Image) -> SceneProfile:
    """Heuristic scene profile. Rain and underwater are not detectable from these statistics;
    object scale and density are unknown and reported as mixed and moderate."""
    lum = mean_luminance(image)
    con = contrast(image)
    dark = dark_channel_strength(image)
    sigma = noise_estimate(image)

    if lum < LOW_LIGHT_LUMINANCE:
        label = "low_light"
    elif sigma > NOISE_SIGMA:
        label = "noise"
    elif dark > FOG_DARK_CHANNEL and con < FOG_CONTRAST and lum > FOG_MIN_LUMINANCE:
        label = "fog"
    else:
        label = "normal"

    illumination = (
        "dark" if lum < 0.15 else "dim" if lum < 0.35 else "normal" if lum < 0.7 else "bright"
    )
    visibility = "poor" if con < 0.1 else "moderate" if con < 0.18 else "clear"
    return SceneProfile(
        scene_label=label,
        illumination=illumination,
        visibility=visibility,
        object_scale="mixed",
        object_density="moderate",
        confidence=HEURISTIC_CONFIDENCE,
    )


def laplacian_variance(image: Image) -> float:
    """Sharpness: variance of the 4-neighbour Laplacian of the luminance (0 for tiny images)."""
    y = _luma(image)
    if y.shape[0] < 3 or y.shape[1] < 3:
        return 0.0
    lap = (
        4.0 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
    )
    return float(lap.var())
