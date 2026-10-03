"""Instance grouping (paper Eq. 1 and 2, anchor-based greedy; system-design 5.2 item 7)."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image as PILImage

from discern.config.settings import GroupingThresholds
from discern.models.roles import Detection, Image
from discern.vision.boxes import Box, clip, expand, iou

Vector = npt.NDArray[np.float32]


@dataclass(frozen=True)
class InstanceGroup:
    anchor: Detection  # highest-score member
    members: tuple[Detection, ...]  # anchor first, then in descending score order

    @property
    def box(self) -> Box:
        """Fused box: the anchor's."""
        return self.anchor.box


def crop_box(image: Image, box: Box, alpha: float) -> Image:
    """Expand by alpha, clip to the image, crop (at least one pixel in each dimension)."""
    height, width = image.shape[:2]
    b = clip(expand(box, alpha), width, height)
    x1 = min(max(math.floor(b.x1), 0), width - 1)
    y1 = min(max(math.floor(b.y1), 0), height - 1)
    x2 = min(max(math.ceil(b.x2), x1 + 1), width)
    y2 = min(max(math.ceil(b.y2), y1 + 1), height)
    return image[y1:y2, x1:x2]


def crop_embedding(image: Image, box: Box, alpha: float, size: int) -> Vector:
    """Eq. 1: expanded crop resized to size x size, flattened and L2-normalized."""
    crop = crop_box(image, box, alpha)
    resized = PILImage.fromarray(crop).resize((size, size), PILImage.Resampling.BILINEAR)
    v = np.asarray(resized, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _similarity(a: Vector, b: Vector) -> float:
    if not a.any() and not b.any():
        return 1.0  # two all-black crops are identical
    return float(np.dot(a, b))


def group_detections(
    image: Image, detections: Sequence[Detection], thresholds: GroupingThresholds
) -> list[InstanceGroup]:
    """Eq. 2: greedy assignment in descending score order against each group's anchor."""
    ordered = sorted(detections, key=lambda d: -d.score)  # stable: ties keep input order
    anchors: list[tuple[Detection, Vector]] = []
    members: list[list[Detection]] = []
    for det in ordered:
        emb = crop_embedding(image, det.box, thresholds.alpha, thresholds.crop_size)
        for i, (anchor, anchor_emb) in enumerate(anchors):
            if (
                iou(det.box, anchor.box) > thresholds.theta_iou
                and _similarity(emb, anchor_emb) > thresholds.phi_vis
            ):
                members[i].append(det)
                break
        else:
            anchors.append((det, emb))
            members.append([det])
    return [InstanceGroup(a, tuple(m)) for (a, _), m in zip(anchors, members, strict=True)]
