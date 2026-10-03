import numpy as np
import pytest

from discern.models.roles import Image
from discern.models.tiling import pad_to_multiple, plan_tiles, run_tiled, upscale_from_x4


def _image(h: int, w: int) -> Image:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_tiles_cover_every_pixel_exactly_once() -> None:
    h, w = 70, 130
    covered = np.zeros((h, w), dtype=int)
    for t in plan_tiles(h, w, tile=32, pad=8):
        y0, y1, x0, x1 = t.core
        covered[y0:y1, x0:x1] += 1
        sy0, sy1, sx0, sx1 = t.src
        assert sy0 <= y0 and y1 <= sy1 and sx0 <= x0 and x1 <= sx1
        assert sy0 >= 0 and sx0 >= 0 and sy1 <= h and sx1 <= w
    assert (covered == 1).all()


@pytest.mark.parametrize("scale", [1, 2, 4])
@pytest.mark.parametrize("shape", [(70, 130), (31, 33), (64, 64), (100, 20)])
def test_tiled_matches_untiled_for_pointwise_function(scale: int, shape: tuple[int, int]) -> None:
    def fn(x: Image) -> Image:
        y = (255 - x).astype(np.uint8)
        return np.repeat(np.repeat(y, scale, axis=0), scale, axis=1)

    img = _image(*shape)
    assert np.array_equal(run_tiled(img, fn, scale, tile=32, pad=8), fn(img))


def test_small_image_is_processed_in_one_call() -> None:
    calls: list[tuple[int, ...]] = []

    def fn(x: Image) -> Image:
        calls.append(x.shape)
        return x

    run_tiled(_image(20, 20), fn, 1, tile=32, pad=8)
    assert len(calls) == 1


@pytest.mark.parametrize("shape", [(17, 33), (16, 16), (1, 5)])
def test_pad_to_multiple_pads_bottom_right_and_reports_original(shape: tuple[int, int]) -> None:
    img = _image(*shape)
    padded, orig = pad_to_multiple(img, 8)
    assert orig == shape
    assert padded.shape[0] % 8 == 0 and padded.shape[1] % 8 == 0
    assert np.array_equal(padded[: shape[0], : shape[1]], img)


def test_upscale_factor_four_returns_model_output_unchanged() -> None:
    img = _image(10, 12)
    out = upscale_from_x4(img, 4, lambda x: np.repeat(np.repeat(x, 4, 0), 4, 1))
    assert out.shape == (40, 48, 3)


def test_upscale_factor_two_downscales_the_x4_output() -> None:
    img = np.full((10, 12, 3), 90, dtype=np.uint8)
    out = upscale_from_x4(img, 2, lambda x: np.repeat(np.repeat(x, 4, 0), 4, 1))
    assert out.shape == (20, 24, 3) and out.dtype == np.uint8
    assert (out == 90).all()


@pytest.mark.parametrize("factor", [1, 3, 8])
def test_unsupported_factor_is_rejected(factor: int) -> None:
    with pytest.raises(ValueError):
        upscale_from_x4(_image(4, 4), factor, lambda x: x)
