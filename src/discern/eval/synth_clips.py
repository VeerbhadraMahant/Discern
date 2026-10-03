"""Synthetic video clips built from labelled still images (system-design 10.2).

A clip is a slow, deterministic zoom from the full image to a centred crop of `end_scale`, every
frame resized back to the image size. Crops are nested (each is centred and smaller than the
last), so a ground-truth object that is inside the final crop is inside every earlier crop and
stays visible for the whole clip. The number of distinct such objects per label is the true count
the video pipeline should report. An optional degradation is applied to every frame with one
seed, through `discern.eval.degrade`.

Pure module: numpy and PIL only; no model, torch or MLflow import.
"""

import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage
from pydantic import BaseModel

from discern.config.settings import DegradeThresholds
from discern.eval.degrade import Kind, degrade_clip
from discern.eval.types import AnnotatedImage, GroundTruthBox
from discern.models.roles import Image
from discern.vision.boxes import Box, intersection

DEFAULT_END_SCALE = 0.85
DEFAULT_MIN_VISIBLE_FRACTION = 0.8  # share of a box's area inside the crop to count as visible


@dataclass(frozen=True)
class Degradation:
    kind: Kind
    severity: float


@dataclass(frozen=True)
class VisibleObject:
    index: int  # position in the ground-truth list the clip was built from
    label: str
    box: Box  # in the coordinates of the output frames (clipped to the frame)


class ClipObject(BaseModel):
    """One ground-truth object of the source image, in source-image pixels."""

    label: str
    box: Box


class ClipEntry(BaseModel):
    """One clip in `manifest.json`. `gt_objects` and the zoom parameters are enough to rebuild
    the per-frame ground truth with `visible_in_crop` and `crop_at`."""

    clip: str  # path relative to the clips directory, with forward slashes
    name: str  # clip set, for example bdd100k_clear_fog
    source_dataset: str
    image_id: str
    degradation: Degradation | None = None
    expected_counts: dict[str, int]
    final_crop: Box
    fps: float
    duration_seconds: float
    n_frames: int
    width: int
    height: int
    end_scale: float
    min_visible_fraction: float
    seed: int
    gt_objects: list[ClipObject]


class Manifest(BaseModel):
    clips: list[ClipEntry]


def clip_seed(image_id: str) -> int:
    """Stable per-clip seed (Python's hash() is salted per process, crc32 is not)."""
    return zlib.crc32(image_id.encode("utf-8"))


def crop_at(
    width: int, height: int, end_scale: float, frame: int, n_frames: int, fps: float
) -> Box:
    """Centred crop of frame `frame`: its size goes linearly from the full frame at t = 0 to
    `end_scale` of it at the last frame."""
    if not 0.0 < end_scale <= 1.0:
        raise ValueError(f"end_scale must be in (0, 1], got {end_scale}")
    if n_frames < 1 or not 0 <= frame < n_frames:
        raise ValueError(f"frame {frame} is outside a clip of {n_frames} frames")
    if fps <= 0:
        raise ValueError("fps must be positive")
    span = (n_frames - 1) / fps
    progress = (frame / fps) / span if span > 0 else 1.0
    scale = 1.0 + (end_scale - 1.0) * progress
    cw, ch = width * scale, height * scale
    x1, y1 = (width - cw) / 2.0, (height - ch) / 2.0
    return Box(x1, y1, x1 + cw, y1 + ch)


def final_crop(width: int, height: int, end_scale: float) -> Box:
    """The crop of the last frame."""
    return crop_at(width, height, end_scale, 1, 2, 1.0)


def visible_fraction(box: Box, crop: Box) -> float:
    """Share of `box`'s area that lies inside `crop`."""
    return intersection(box, crop).area / box.area if box.area > 0 else 0.0


def expected_counts(
    gt_boxes: Sequence[GroundTruthBox],
    crop: Box,
    min_visible_fraction: float = DEFAULT_MIN_VISIBLE_FRACTION,
) -> dict[str, int]:
    """Distinct ground-truth objects per label that are at least `min_visible_fraction` inside
    `crop`. Each ground-truth box is one distinct object. Labels with no such object are absent."""
    counts: dict[str, int] = {}
    for gt in gt_boxes:
        if visible_fraction(gt.box, crop) >= min_visible_fraction:
            counts[gt.label] = counts.get(gt.label, 0) + 1
    return counts


def visible_in_crop(
    gt_boxes: Sequence[GroundTruthBox],
    crop: Box,
    out_size: tuple[int, int],
    min_visible_fraction: float = DEFAULT_MIN_VISIBLE_FRACTION,
) -> list[VisibleObject]:
    """Objects at least `min_visible_fraction` inside `crop`, with their boxes clipped to the crop
    and mapped to an output frame of `out_size` (width, height)."""
    sx, sy = out_size[0] / (crop.x2 - crop.x1), out_size[1] / (crop.y2 - crop.y1)
    out: list[VisibleObject] = []
    for i, gt in enumerate(gt_boxes):
        if visible_fraction(gt.box, crop) < min_visible_fraction:
            continue
        clipped = intersection(gt.box, crop)
        box = Box(
            (clipped.x1 - crop.x1) * sx,
            (clipped.y1 - crop.y1) * sy,
            (clipped.x2 - crop.x1) * sx,
            (clipped.y2 - crop.y1) * sy,
        )
        out.append(VisibleObject(i, gt.label, box))
    return out


def make_zoom_clip(
    image_rgb: Image,
    gt_boxes: Sequence[GroundTruthBox],
    n_frames: int,
    fps: float,
    end_scale: float = DEFAULT_END_SCALE,
    *,
    min_visible_fraction: float = DEFAULT_MIN_VISIBLE_FRACTION,
    degradation: Degradation | None = None,
    cfg: DegradeThresholds | None = None,
    seed: int = 0,
) -> tuple[list[Image], list[list[VisibleObject]]]:
    """Frames of the zoom clip and, per frame, the ground-truth objects visible in it.

    The frames are RGB uint8 at the size of `image_rgb`. With `degradation`, `cfg` is required
    and every frame is degraded with the same `seed` (one fog depth, one rain layout)."""
    if n_frames < 1:
        raise ValueError("n_frames must be at least 1")
    if degradation is not None and cfg is None:
        raise ValueError("a degradation needs the degrade thresholds (cfg)")
    height, width = image_rgb.shape[:2]
    source = PILImage.fromarray(image_rgb)
    frames: list[Image] = []
    visible: list[list[VisibleObject]] = []
    for i in range(n_frames):
        crop = crop_at(width, height, end_scale, i, n_frames, fps)
        resized = source.resize(
            (width, height),
            PILImage.Resampling.BILINEAR,
            box=(crop.x1, crop.y1, crop.x2, crop.y2),
        )
        frames.append(np.asarray(resized, dtype=np.uint8))
        visible.append(visible_in_crop(gt_boxes, crop, (width, height), min_visible_fraction))
    if degradation is not None and cfg is not None:
        frames = degrade_clip(frames, degradation.kind, degradation.severity, cfg, seed)
    return frames, visible


def select_images(
    images: Sequence[AnnotatedImage],
    n: int,
    *,
    min_objects: int = 3,
    max_objects: int = 25,
    end_scale: float = DEFAULT_END_SCALE,
    min_visible_fraction: float = DEFAULT_MIN_VISIBLE_FRACTION,
) -> list[AnnotatedImage]:
    """The first `n` images by sorted id with `min_objects` to `max_objects` labelled objects and
    at least one object inside the final crop (so some count is at least 1)."""
    chosen: list[AnnotatedImage] = []
    for a in sorted(images, key=lambda a: a.image_id):
        if not min_objects <= len(a.objects) <= max_objects:
            continue
        if not expected_counts(
            a.objects, final_crop(a.width, a.height, end_scale), min_visible_fraction
        ):
            continue
        chosen.append(a)
        if len(chosen) == n:
            break
    return chosen
