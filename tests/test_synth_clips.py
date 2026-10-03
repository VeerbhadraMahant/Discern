from pathlib import Path

import numpy as np
import pytest

from discern.config import load_settings
from discern.eval.synth_clips import (
    Degradation,
    clip_seed,
    crop_at,
    expected_counts,
    final_crop,
    make_zoom_clip,
    select_images,
    visible_fraction,
    visible_in_crop,
)
from discern.eval.types import AnnotatedImage, GroundTruthBox
from discern.models.roles import Image
from discern.vision.boxes import Box

CFG = load_settings("local_lite").thresholds.degrade
W, H = 100, 80
# end_scale 0.5 on 100x80: final crop (25, 20, 75, 60)
GT = [
    GroundTruthBox(box=Box(30, 25, 50, 45), label="car"),  # fully inside
    GroundTruthBox(box=Box(0, 0, 20, 20), label="car"),  # outside the final crop
    GroundTruthBox(box=Box(60, 50, 90, 70), label="car"),  # 15x10 of 30x20 inside: 0.25
    GroundTruthBox(box=Box(40, 30, 60, 50), label="person"),  # fully inside
]


def _image() -> Image:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, size=(H, W, 3)).astype(np.uint8)


def test_crop_schedule() -> None:
    assert crop_at(W, H, 0.5, 0, 3, 10.0) == Box(0, 0, 100, 80)
    assert crop_at(W, H, 0.5, 1, 3, 10.0) == Box(12.5, 10, 87.5, 70)
    assert crop_at(W, H, 0.5, 2, 3, 10.0) == Box(25, 20, 75, 60) == final_crop(W, H, 0.5)


def test_crops_are_nested() -> None:
    crops = [crop_at(W, H, 0.7, i, 9, 10.0) for i in range(9)]
    for outer, inner in zip(crops, crops[1:], strict=False):
        assert outer.x1 <= inner.x1 and outer.y1 <= inner.y1
        assert outer.x2 >= inner.x2 and outer.y2 >= inner.y2


def test_bad_arguments() -> None:
    with pytest.raises(ValueError):
        crop_at(W, H, 1.2, 0, 3, 10.0)
    with pytest.raises(ValueError):
        crop_at(W, H, 0.5, 3, 3, 10.0)


def test_visible_fraction() -> None:
    crop = final_crop(W, H, 0.5)
    assert visible_fraction(GT[0].box, crop) == 1.0
    assert visible_fraction(GT[1].box, crop) == 0.0
    assert visible_fraction(GT[2].box, crop) == pytest.approx(0.25)


def test_expected_counts_by_fraction() -> None:
    crop = final_crop(W, H, 0.5)
    assert expected_counts(GT, crop, 0.8) == {"car": 1, "person": 1}
    assert expected_counts(GT, crop, 0.2) == {"car": 2, "person": 1}


def test_visible_boxes_are_mapped_to_frame_coordinates() -> None:
    crop = final_crop(W, H, 0.5)
    seen = visible_in_crop(GT, crop, (W, H), 0.8)
    assert [(v.index, v.label) for v in seen] == [(0, "car"), (3, "person")]
    assert seen[0].box == Box(10, 10, 50, 50)  # (30-25)*2, (25-20)*2, (50-25)*2, (45-20)*2


def test_clip_shapes_and_visibility() -> None:
    frames, visible = make_zoom_clip(_image(), GT, 5, 10.0, 0.5, min_visible_fraction=0.8)
    assert len(frames) == len(visible) == 5
    assert all(f.shape == (H, W, 3) and f.dtype == np.uint8 for f in frames)
    assert [v.index for v in visible[0]] == [0, 1, 2, 3]  # the full frame shows everything
    assert [v.index for v in visible[-1]] == [0, 3]
    final = {v.index for v in visible[-1]}
    assert all(final <= {v.index for v in frame} for frame in visible)  # never disappear
    assert np.abs(frames[0].astype(int) - _image().astype(int)).max() <= 1


def test_zoom_is_deterministic_and_zooms() -> None:
    a, _ = make_zoom_clip(_image(), GT, 4, 10.0, 0.5)
    b, _ = make_zoom_clip(_image(), GT, 4, 10.0, 0.5)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not np.array_equal(a[0], a[-1])


def test_degradation_is_seeded_and_leaves_ground_truth_alone() -> None:
    fog = Degradation("fog", 0.6)
    clean, _ = make_zoom_clip(_image(), GT, 3, 10.0, 0.5)
    a, va = make_zoom_clip(_image(), GT, 3, 10.0, 0.5, degradation=fog, cfg=CFG, seed=5)
    b, _ = make_zoom_clip(_image(), GT, 3, 10.0, 0.5, degradation=fog, cfg=CFG, seed=5)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert all(not np.array_equal(x, y) for x, y in zip(a, clean, strict=True))
    assert [v.index for v in va[-1]] == [0, 3]


def test_noise_differs_from_frame_to_frame() -> None:
    flat = np.full((H, W, 3), 128, dtype=np.uint8)
    noisy, _ = make_zoom_clip(
        flat, GT, 3, 10.0, 0.5, degradation=Degradation("noise", 0.5), cfg=CFG, seed=5
    )
    assert not np.array_equal(noisy[0], noisy[1])


def test_degradation_needs_cfg() -> None:
    with pytest.raises(ValueError):
        make_zoom_clip(_image(), GT, 3, 10.0, degradation=Degradation("fog", 0.5))


def test_clip_seed_is_stable() -> None:
    assert clip_seed("abc") == clip_seed("abc") != clip_seed("abd")
    assert clip_seed("abc") == 891568578  # crc32 of b"abc"


def _annotated(image_id: str, boxes: list[GroundTruthBox]) -> AnnotatedImage:
    return AnnotatedImage(
        image_id=image_id, path=Path("x.jpg"), width=W, height=H, objects=tuple(boxes)
    )


def test_select_images_filters_and_sorts() -> None:
    inside = [GroundTruthBox(box=Box(30, 25, 50, 45), label="car")] * 3
    outside = [GroundTruthBox(box=Box(0, 0, 10, 10), label="car")] * 5
    images = [
        _annotated("d", inside),
        _annotated("a", inside[:2]),  # too few objects
        _annotated("c", inside * 9),  # 27 objects: too many
        _annotated("b", inside),
        _annotated("e", outside),  # nothing in the final crop
        _annotated("f", inside),
    ]
    picked = select_images(images, 2, end_scale=0.5)
    assert [a.image_id for a in picked] == ["b", "d"]
