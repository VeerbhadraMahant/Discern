"""Haze removal for display (the 'clear view'). Pure numpy and OpenCV, no weights.

Dark channel prior with a guided-filter transmission, then a shared auto-levels stretch and an
exposure correction that keeps the mean brightness near the input's, so the result is clear and
not dark.
Detection never sees this image unless the agent chose a restored image; it only changes what the
user is shown.
"""

import cv2
import numpy as np

from discern.models.roles import Image

OMEGA = 0.9  # how much haze to remove (1.0 removes all, which looks unnatural)
PATCH = 15
AIRLIGHT_FRACTION = 0.001  # brightest share of the dark channel used to estimate airlight
T_MIN = 0.1  # transmission floor, limits noise amplification in dense haze
GUIDE_RADIUS = 60
GUIDE_EPS = 1e-3
LEVEL_PERCENTILES = (1.0, 99.0)
CHROMA_KEEP = 0.55  # share of the recovered colour differences kept, to avoid over-saturation
MAX_GAIN = 1.8  # cap on the exposure correction


def _box(x: np.ndarray, radius: int) -> np.ndarray:
    out: np.ndarray = cv2.boxFilter(x, -1, (radius, radius))
    return out


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    mean_g, mean_s = _box(guide, radius), _box(src, radius)
    var_g = _box(guide * guide, radius) - mean_g * mean_g
    a = (_box(guide * src, radius) - mean_g * mean_s) / (var_g + eps)
    b = mean_s - a * mean_g
    out: np.ndarray = _box(a, radius) * guide + _box(b, radius)
    return out


def clear_view(image: Image) -> Image:
    """Return a haze-free copy of an RGB uint8 image with the same shape."""
    rgb = image.astype(np.float64) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (PATCH, PATCH))
    dark = cv2.erode(rgb.min(axis=2), kernel)
    count = max(int(dark.size * AIRLIGHT_FRACTION), 1)
    brightest = np.argsort(dark.ravel())[-count:]
    # One grey airlight value: haze is close to neutral, and a per-channel estimate taken from a few
    # bright pixels pushes the whole image toward the colour of those pixels.
    airlight = np.full(3, max(float(rgb.reshape(-1, 3)[brightest].mean()), 1e-3))

    transmission = 1.0 - OMEGA * cv2.erode((rgb / airlight).min(axis=2), kernel)
    grey = rgb.mean(axis=2)
    transmission = _guided_filter(grey, transmission, GUIDE_RADIUS, GUIDE_EPS)
    transmission = np.clip(transmission, T_MIN, 1.0)

    recovered = np.clip((rgb - airlight) / transmission[..., None] + airlight, 0.0, 1.0)
    # Recovery multiplies colour differences by 1/t, which over-saturates; keep only part of that.
    grey_out = recovered.mean(axis=2, keepdims=True)
    recovered = np.clip(grey_out + CHROMA_KEEP * (recovered - grey_out), 0.0, 1.0)

    # One shared stretch for all channels: stretching each channel alone shifts the colours.
    low, high = np.percentile(recovered.mean(axis=2), LEVEL_PERCENTILES)
    recovered = np.clip((recovered - low) / max(high - low, 1e-3), 0.0, 1.0)

    gain = float(np.clip(rgb.mean() / max(recovered.mean(), 1e-3), 1.0, MAX_GAIN))
    recovered = np.clip(recovered * gain, 0.0, 1.0)
    out: Image = np.rint(recovered * 255.0).astype(np.uint8)
    return out
