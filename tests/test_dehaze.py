import numpy as np

from discern.config.settings import load_settings
from discern.eval.degrade import fog
from discern.vision.dehaze import clear_view
from discern.vision.stats import contrast, dark_channel_strength, mean_luminance


def _scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.integers(20, 200, size=(96, 128, 3), dtype=np.uint8)
    base[:, :64] = (base[:, :64] // 2) + 100
    return base


def test_clear_view_removes_haze_and_keeps_shape_and_brightness() -> None:
    cfg = load_settings().thresholds.degrade
    hazy = fog(_scene(), 0.8, cfg)
    out = clear_view(hazy)
    assert out.shape == hazy.shape and out.dtype == np.uint8
    assert contrast(out) > contrast(hazy)
    assert dark_channel_strength(out) < dark_channel_strength(hazy)
    assert abs(mean_luminance(out) - mean_luminance(hazy)) < 0.25


def test_clear_view_is_deterministic() -> None:
    img = _scene()
    assert np.array_equal(clear_view(img), clear_view(img))
