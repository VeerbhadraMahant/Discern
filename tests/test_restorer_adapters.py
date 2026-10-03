from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from discern.models.loading import load_adapter
from discern.models.registry import load_registry
from discern.models.roles import Image

REGISTRY = load_registry()
ROOT = Path(__file__).resolve().parents[1]
FOG = sorted((ROOT / "data" / "hazydet_real" / "images").glob("*.jpg"))
NIGHT = sorted((ROOT / "data" / "bdd100k_night" / "images").glob("*.jpg"))


def _load(paths: list[Path]) -> Image:
    if not paths:
        pytest.skip("dataset images not available")
    return np.asarray(PILImage.open(paths[0]).convert("RGB"))[:360, :640].copy()


def _check_restored(src: Image, out: Image) -> None:
    assert out.shape == src.shape and out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    assert not np.array_equal(out, src)


def test_classical_adapters_follow_the_restorer_protocol() -> None:
    pytest.importorskip("cv2")
    fog, night = _load(FOG), _load(NIGHT)
    for name, src in [
        ("classical-dehaze", fog),
        ("classical-lowlight", night),
        ("classical-denoise", night),
    ]:
        adapter = load_adapter(REGISTRY[name], device="cpu")
        assert adapter.name == name  # type: ignore[attr-defined]
        _check_restored(src, adapter.restore(src))  # type: ignore[attr-defined]


def test_classical_lowlight_brightens_a_dark_image() -> None:
    pytest.importorskip("cv2")
    night = _load(NIGHT)
    adapter = load_adapter(REGISTRY["classical-lowlight"], device="cpu")
    assert adapter.restore(night).mean() > night.mean()  # type: ignore[attr-defined]


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("swinir-denoise", NIGHT),
        ("zero-dce-pp", NIGHT),
        ("llflow-lowlight", NIGHT),
        ("mprnet-derain", FOG),
    ],
)
def test_restorer_smoke(name: str, source: list[Path]) -> None:
    pytest.importorskip("torch")
    src = _load(source)
    adapter = load_adapter(REGISTRY[name])
    _check_restored(src, adapter.restore(src))  # type: ignore[attr-defined]


@pytest.mark.gpu
@pytest.mark.parametrize("factor", [2, 4])
def test_real_esrgan_smoke(factor: int) -> None:
    pytest.importorskip("torch")
    src = _load(FOG)[:120, :160].copy()
    out = load_adapter(REGISTRY["real-esrgan-x4plus"]).upscale(src, factor)  # type: ignore[attr-defined]
    assert out.shape == (120 * factor, 160 * factor, 3) and out.dtype == np.uint8
