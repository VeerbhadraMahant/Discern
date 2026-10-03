"""Deterministic synthetic degradation of RGB uint8 frames and clips (system-design 10.2).

Severity is in [0, 1]; 0 returns the frame unchanged. Parameter ranges come from the `degrade`
section of thresholds.yaml. Everything is seeded: the same frame, severity and seed give the
same pixels. Clips share one fog depth and airlight and one rain seed, so only the noise
differs from frame to frame while rain streaks fall.
"""

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from discern.config.settings import DegradeThresholds
from discern.models.roles import Image

Kind = Literal["fog", "low_light", "rain", "noise"]


def _lerp(span: tuple[float, float], severity: float) -> float:
    return span[0] + (span[1] - span[0]) * severity


def _check(severity: float) -> float:
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    return severity


def _to_u8(x: np.ndarray) -> Image:
    out: Image = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    return out


def depth_prior(height: int, width: int) -> np.ndarray:
    """Normalised depth for road-like scenes: 1 at the top row (far), 0 at the bottom (near)."""
    rows = 1.0 - np.arange(height, dtype=np.float64) / max(height - 1, 1)
    return np.repeat(rows[:, None], width, axis=1)


def fog(
    image: Image, severity: float, cfg: DegradeThresholds, depth: np.ndarray | None = None
) -> Image:
    """Atmospheric scattering I = J * t + A * (1 - t), t = exp(-beta * depth).

    `depth` is an HxW map in 0..1 (far is 1); without one, a vertical-gradient prior is used.
    """
    if _check(severity) == 0.0:
        return image.copy()
    h, w = image.shape[:2]
    depth_map = depth_prior(h, w) if depth is None else depth
    if depth_map.shape != (h, w):
        raise ValueError(f"depth map is {depth_map.shape}, image is {(h, w)}")
    beta = _lerp(cfg.fog_beta, severity)
    airlight = _lerp(cfg.fog_airlight, severity) * 255.0
    t = np.exp(-beta * depth_map)[..., None]
    return _to_u8(image.astype(np.float64) * t + airlight * (1.0 - t))


def low_light(
    image: Image, severity: float, cfg: DegradeThresholds, rng: np.random.Generator
) -> Image:
    """Gamma darkening, exposure scaling, then shot (poisson) noise."""
    if _check(severity) == 0.0:
        return image.copy()
    gamma = _lerp(cfg.low_light_gamma, severity)
    scale = _lerp(cfg.low_light_scale, severity)
    peak = _lerp(cfg.low_light_peak, severity)
    dark = scale * (image.astype(np.float64) / 255.0) ** gamma
    noisy = rng.poisson(dark * peak) / peak
    return _to_u8(noisy * 255.0)


def sensor_noise(
    image: Image, severity: float, cfg: DegradeThresholds, rng: np.random.Generator
) -> Image:
    """Poisson (signal-dependent) plus gaussian (read) noise."""
    if _check(severity) == 0.0:
        return image.copy()
    sigma = _lerp(cfg.noise_sigma, severity)
    peak = _lerp(cfg.noise_peak, severity)
    x = image.astype(np.float64)
    shot = rng.poisson(x / 255.0 * peak) / peak * 255.0
    return _to_u8(shot + rng.normal(0.0, sigma, size=x.shape))


def rain(
    image: Image, severity: float, cfg: DegradeThresholds, seed: int, frame_index: int = 0
) -> Image:
    """Overlay seeded streaks. The streak layout depends on `seed` only; `frame_index` moves
    every streak down by `rain_fall_fraction` of the frame height per frame."""
    if _check(severity) == 0.0:
        return image.copy()
    h, w = image.shape[:2]
    rng = np.random.default_rng(seed)
    count = int(round(_lerp(cfg.rain_density, severity) * h * w))
    length = max(2.0, _lerp(cfg.rain_length, severity) * h)
    brightness = _lerp(cfg.rain_brightness, severity)
    angle = math.radians(cfg.rain_angle_degrees)
    dx, dy = -math.sin(angle) * length, math.cos(angle) * length
    xs = rng.uniform(0.0, w, size=count)
    ys = rng.uniform(0.0, h, size=count)
    gains = rng.uniform(0.5, 1.0, size=count)
    fall = cfg.rain_fall_fraction * h * frame_index
    layer = PILImage.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layer)
    for x, y, gain in zip(xs, ys, gains, strict=True):
        y0 = (y + fall) % h
        for shift in (0.0, -float(h)):  # wrapped copy so streaks cross the bottom edge
            draw.line(
                [(x, y0 + shift), (x + dx, y0 + shift + dy)], fill=int(255 * gain), width=1
            )
    alpha = (np.asarray(layer, dtype=np.float64) / 255.0 * brightness)[..., None]
    return _to_u8(image.astype(np.float64) * (1.0 - alpha) + 255.0 * alpha)


def degrade_frame(
    image: Image,
    kind: Kind,
    severity: float,
    cfg: DegradeThresholds,
    seed: int = 0,
    frame_index: int = 0,
    depth: np.ndarray | None = None,
) -> Image:
    """Degrade one frame. `frame_index` selects the noise draw and the rain position."""
    rng = np.random.default_rng([seed, frame_index])
    if kind == "fog":
        return fog(image, severity, cfg, depth)
    if kind == "low_light":
        return low_light(image, severity, cfg, rng)
    if kind == "rain":
        return rain(image, severity, cfg, seed, frame_index)
    if kind == "noise":
        return sensor_noise(image, severity, cfg, rng)
    raise ValueError(f"unknown degradation {kind!r}")


def degrade_clip(
    frames: Sequence[Image],
    kind: Kind,
    severity: float,
    cfg: DegradeThresholds,
    seed: int = 0,
    depth: np.ndarray | None = None,
) -> list[Image]:
    """Degrade a clip with one fog depth and airlight and one rain layout for every frame."""
    return [degrade_frame(f, kind, severity, cfg, seed, i, depth) for i, f in enumerate(frames)]
