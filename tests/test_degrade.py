import numpy as np
import pytest

from discern.config import load_settings
from discern.eval.degrade import (
    Kind,
    degrade_clip,
    degrade_frame,
    depth_prior,
    fog,
    low_light,
    rain,
)
from discern.models.roles import Image

CFG = load_settings("local_lite").thresholds.degrade
KINDS: list[Kind] = ["fog", "low_light", "rain", "noise"]


def _scene(h: int = 48, w: int = 64, seed: int = 0) -> Image:
    rng = np.random.default_rng(seed)
    return rng.integers(20, 236, size=(h, w, 3)).astype(np.uint8)


def _luma(img: Image) -> float:
    return float(img.astype(np.float64).mean())


def _contrast(img: Image) -> float:
    return float(img.astype(np.float64).std())


@pytest.mark.parametrize("kind", KINDS)
def test_dtype_and_shape_preserved(kind: Kind) -> None:
    img = _scene()
    out = degrade_frame(img, kind, 0.7, CFG, seed=3)
    assert out.dtype == np.uint8 and out.shape == img.shape


@pytest.mark.parametrize("kind", KINDS)
def test_severity_zero_is_identity_and_a_copy(kind: Kind) -> None:
    img = _scene()
    out = degrade_frame(img, kind, 0.0, CFG, seed=3)
    assert np.array_equal(out, img)
    assert out is not img


@pytest.mark.parametrize("kind", KINDS)
def test_deterministic_by_seed(kind: Kind) -> None:
    img = _scene()
    a = degrade_frame(img, kind, 0.8, CFG, seed=5, frame_index=2)
    b = degrade_frame(img, kind, 0.8, CFG, seed=5, frame_index=2)
    assert np.array_equal(a, b)
    if kind != "fog":  # fog has no random component
        assert not np.array_equal(a, degrade_frame(img, kind, 0.8, CFG, seed=6, frame_index=2))


def test_severity_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        degrade_frame(_scene(), "fog", 1.5, CFG)


def test_stronger_beta_lowers_contrast() -> None:
    img = _scene()
    contrasts = [_contrast(fog(img, s, CFG)) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert contrasts == sorted(contrasts, reverse=True)
    assert contrasts[-1] < contrasts[0]


def test_fog_matches_scattering_model_on_a_flat_image() -> None:
    img = np.full((10, 10, 3), 100, dtype=np.uint8)
    depth = np.ones((10, 10))
    out = fog(img, 1.0, CFG, depth)
    t = np.exp(-CFG.fog_beta[1])
    expected = round(100 * t + CFG.fog_airlight[1] * 255 * (1 - t))
    assert int(out[0, 0, 0]) == expected


def test_fog_prior_is_clear_at_the_bottom_and_thick_at_the_top() -> None:
    depth = depth_prior(10, 4)
    assert depth[0, 0] == 1.0 and depth[-1, 0] == 0.0
    img = np.full((10, 4, 3), 60, dtype=np.uint8)
    out = fog(img, 1.0, CFG)
    assert int(out[-1, 0, 0]) == 60
    assert int(out[0, 0, 0]) > 60


def test_fog_rejects_wrong_depth_shape() -> None:
    with pytest.raises(ValueError):
        fog(_scene(), 0.5, CFG, np.zeros((3, 3)))


def test_low_light_lowers_mean_luminance_monotonically() -> None:
    img = _scene()
    means = [
        _luma(low_light(img, s, CFG, np.random.default_rng(1))) for s in (0.0, 0.3, 0.6, 1.0)
    ]
    assert means == sorted(means, reverse=True)
    assert means[-1] < 0.5 * means[0]


def test_noise_adds_variance_to_a_flat_image() -> None:
    flat = np.full((48, 64, 3), 128, dtype=np.uint8)
    out = degrade_frame(flat, "noise", 1.0, CFG, seed=1)
    assert _contrast(out) > 5.0
    mild = degrade_frame(flat, "noise", 0.2, CFG, seed=1)
    assert _contrast(mild) < _contrast(out)


def test_rain_brightens_and_denser_rain_changes_more() -> None:
    img = np.full((96, 128, 3), 60, dtype=np.uint8)
    light = rain(img, 0.3, CFG, seed=2)
    heavy = rain(img, 1.0, CFG, seed=2)
    assert _luma(light) > 60.0
    assert _luma(heavy) > _luma(light)
    assert int(heavy.min()) >= 60  # streaks only brighten


def test_rain_streaks_move_between_frames_but_share_a_layout() -> None:
    img = np.full((96, 128, 3), 60, dtype=np.uint8)
    f0, f1 = rain(img, 1.0, CFG, 4, 0), rain(img, 1.0, CFG, 4, 1)
    assert not np.array_equal(f0, f1)
    assert abs(_luma(f0) - _luma(f1)) < 0.05 * _luma(f0)


def test_clip_fog_is_frame_consistent() -> None:
    img = _scene()
    out = degrade_clip([img, img, img], "fog", 0.9, CFG, seed=1)
    assert np.array_equal(out[0], out[1]) and np.array_equal(out[1], out[2])


def test_clip_noise_differs_per_frame_but_clip_is_reproducible() -> None:
    img = _scene()
    a = degrade_clip([img, img], "noise", 0.9, CFG, seed=1)
    b = degrade_clip([img, img], "noise", 0.9, CFG, seed=1)
    assert not np.array_equal(a[0], a[1])
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_config_ranges_are_ordered_sensibly() -> None:
    assert CFG.fog_beta[0] == 0.0 < CFG.fog_beta[1]
    assert CFG.low_light_scale[1] < CFG.low_light_scale[0] <= 1.0
    assert CFG.low_light_gamma[1] > CFG.low_light_gamma[0] >= 1.0
    assert 0.0 < CFG.fog_airlight[1] <= 1.0
