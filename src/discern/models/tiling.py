"""Pure numpy helpers shared by the restoration adapters: tiling, padding, factor handling.

Kept free of torch so they are unit-testable on CPU-only CI.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage

from discern.models.roles import Image

Window = tuple[int, int, int, int]  # y0, y1, x0, x1


@dataclass(frozen=True)
class Tile:
    src: Window  # input crop including the context margin, clipped to the image
    core: Window  # the region of the image this tile is responsible for


def _spans(length: int, tile: int) -> list[tuple[int, int]]:
    return [(start, min(start + tile, length)) for start in range(0, length, tile)]


def plan_tiles(h: int, w: int, tile: int, pad: int) -> list[Tile]:
    """Cover an h x w image with core tiles of side `tile`, each with `pad` pixels of context."""
    if tile <= 0 or pad < 0:
        raise ValueError("tile must be positive and pad non-negative")
    return [
        Tile(
            src=(max(y0 - pad, 0), min(y1 + pad, h), max(x0 - pad, 0), min(x1 + pad, w)),
            core=(y0, y1, x0, x1),
        )
        for y0, y1 in _spans(h, tile)
        for x0, x1 in _spans(w, tile)
    ]


def run_tiled(image: Image, fn: Callable[[Image], Image], scale: int, tile: int, pad: int) -> Image:
    """Apply `fn` (output is `scale` times larger) tile by tile and stitch the core regions."""
    h, w = image.shape[:2]
    if h <= tile and w <= tile:
        return fn(image)
    out = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    for t in plan_tiles(h, w, tile, pad):
        sy0, sy1, sx0, sx1 = t.src
        cy0, cy1, cx0, cx1 = t.core
        result = fn(image[sy0:sy1, sx0:sx1])
        oy, ox = (cy0 - sy0) * scale, (cx0 - sx0) * scale
        out[cy0 * scale : cy1 * scale, cx0 * scale : cx1 * scale] = result[
            oy : oy + (cy1 - cy0) * scale, ox : ox + (cx1 - cx0) * scale
        ]
    return out


def pad_to_multiple(image: Image, multiple: int) -> tuple[Image, tuple[int, int]]:
    """Reflect-pad bottom and right so both sides are multiples of `multiple`.

    Returns the padded image and the original (h, w) to crop back to.
    """
    h, w = image.shape[:2]
    pad_h, pad_w = -h % multiple, -w % multiple
    if not (pad_h or pad_w):
        return image, (h, w)
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="symmetric")
    return padded, (h, w)


SR_FACTORS = (2, 4)


def upscale_from_x4(image: Image, factor: int, x4: Callable[[Image], Image]) -> Image:
    """Super-resolve by 2 or 4 using a x4 model; 2 is the x4 output downscaled by half."""
    if factor not in SR_FACTORS:
        raise ValueError(f"unsupported SR factor {factor}; use one of {SR_FACTORS}")
    big = x4(image)
    if factor == 4:
        return big
    h, w = image.shape[:2]
    resized = PILImage.fromarray(big).resize((w * 2, h * 2), PILImage.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)
