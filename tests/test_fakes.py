import numpy as np
import pytest

from discern.models.fakes import (
    FakeDetector,
    FakeEmbedder,
    FakeRestorer,
    FakeSuperResolver,
    FakeVLM,
)
from discern.models.roles import VLM, Detection, Detector, Embedder, Restorer, SuperResolver
from discern.vision.boxes import Box

IMG = np.zeros((4, 6, 3), dtype=np.uint8)


def test_fakes_satisfy_role_protocols() -> None:
    _: tuple[VLM, Detector, Restorer, SuperResolver, Embedder] = (
        FakeVLM([]),
        FakeDetector("d"),
        FakeRestorer(),
        FakeSuperResolver(),
        FakeEmbedder(),
    )


def test_fake_vlm_returns_scripted_responses_in_order() -> None:
    vlm = FakeVLM(["a", "b"])
    assert vlm.generate("p1") == "a"
    assert vlm.generate("p2") == "b"
    assert vlm.prompts == ["p1", "p2"]
    with pytest.raises(AssertionError):
        vlm.generate("p3")


def test_fake_detector_filters_by_target() -> None:
    det = Detection(box=Box(0, 0, 1, 1), label="car", score=0.9, detector="d")
    assert FakeDetector("d", [det]).detect(IMG, ["car"]) == [det]
    assert FakeDetector("d", [det]).detect(IMG, ["person"]) == []


def test_fake_restorer_and_sr_change_pixels_predictably() -> None:
    assert FakeRestorer(offset=10).restore(IMG).max() == 10
    assert FakeSuperResolver().upscale(IMG, 2).shape == (8, 12, 3)


def test_fake_embedder_is_deterministic_and_unit_norm() -> None:
    emb = FakeEmbedder()
    a = emb.embed_text(["x"])
    assert np.allclose(a, emb.embed_text(["x"]))
    assert np.isclose(np.linalg.norm(a[0]), 1.0)
