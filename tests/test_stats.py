import numpy as np

from discern.models.roles import Image
from discern.vision.stats import (
    contrast,
    dark_channel_strength,
    mean_luminance,
    noise_estimate,
    profile_from_stats,
)


def textured(seed: int = 0, size: int = 96) -> Image:
    """Smooth random image with dark and bright regions and no noise."""
    rng = np.random.default_rng(seed)
    coarse = rng.uniform(0, 255, size=(6, 6, 3))
    big = np.kron(coarse, np.ones((size // 6, size // 6, 1)))
    return big.astype(np.uint8)


def test_normal_image() -> None:
    profile = profile_from_stats(textured())
    assert profile.scene_label == "normal"
    assert profile.key.startswith("normal|normal|")
    assert profile.key.endswith("|mixed|moderate")


def test_dark_image_is_low_light() -> None:
    profile = profile_from_stats((textured() * 0.1).astype(np.uint8))
    assert profile.scene_label == "low_light"
    assert profile.illumination == "dark"


def test_hazy_image_is_fog() -> None:
    hazy = (textured().astype(np.float64) * 0.2 + 0.7 * 255).astype(np.uint8)
    profile = profile_from_stats(hazy)
    assert profile.scene_label == "fog"
    assert profile.visibility == "poor"


def test_noisy_image_is_noise() -> None:
    rng = np.random.default_rng(1)
    noisy = np.clip(textured().astype(np.float64) + rng.normal(0, 25, (96, 96, 3)), 0, 255)
    assert profile_from_stats(noisy.astype(np.uint8)).scene_label == "noise"


def test_statistics_values() -> None:
    gray = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert abs(mean_luminance(gray) - 128 / 255) < 1e-6
    assert contrast(gray) == 0.0
    assert noise_estimate(gray) == 0.0
    assert abs(dark_channel_strength(gray) - 128 / 255) < 1e-6
