"""Pure box geometry. Boxes are absolute-pixel xyxy floats: (x1, y1, x2, y2)."""

from typing import NamedTuple


class Box(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def intersection(a: Box, b: Box) -> Box:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return Box(0.0, 0.0, 0.0, 0.0)
    return Box(x1, y1, x2, y2)


def expand(box: Box, alpha: float) -> Box:
    """Grow each side outward by alpha times the box's width/height (paper: alpha = 0.25)."""
    dx, dy = box.width * alpha, box.height * alpha
    return Box(box.x1 - dx, box.y1 - dy, box.x2 + dx, box.y2 + dy)


def clip(box: Box, width: float, height: float) -> Box:
    return Box(
        min(max(box.x1, 0.0), width),
        min(max(box.y1, 0.0), height),
        min(max(box.x2, 0.0), width),
        min(max(box.y2, 0.0), height),
    )


def scale(box: Box, sx: float, sy: float | None = None) -> Box:
    """Scale between coordinate spaces, e.g. processing resolution to original frame."""
    sy = sx if sy is None else sy
    return Box(box.x1 * sx, box.y1 * sy, box.x2 * sx, box.y2 * sy)


def union_box(boxes: list[Box]) -> Box:
    if not boxes:
        raise ValueError("union_box needs at least one box")
    return Box(
        min(b.x1 for b in boxes),
        min(b.y1 for b in boxes),
        max(b.x2 for b in boxes),
        max(b.y2 for b in boxes),
    )
