"""Parse a VLM grounding reply into detections. Pure: no torch, no model access.

Qwen3-VL emits boxes as [x1, y1, x2, y2] relative to a 0-1000 grid. This is the one place
those are converted to absolute pixels.
"""

import json
import re
from collections.abc import Sequence

from discern.models.roles import Detection
from discern.vision.boxes import Box, clip

REL_SCALE = 1000.0
_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)\s*```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{[^{}]*\}")


def _payload(text: str) -> object:
    m = _FENCE_RE.search(text)
    body = m["body"] if m else text.strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    # Replies cut off at the token limit: keep the objects that were completed.
    items: list[object] = []
    for m in _OBJECT_RE.finditer(body):
        try:
            items.append(json.loads(m[0]))
        except json.JSONDecodeError:
            continue
    return items


def parse_grounding(
    text: str, width: int, height: int, targets: Sequence[str], detector: str
) -> list[Detection]:
    """Detections from a JSON list of {"bbox_2d": [x1,y1,x2,y2], "label": ...}.

    Invalid JSON gives an empty list (a truncated list keeps its completed objects). Entries
    with unknown labels or malformed boxes are dropped. The VLM gives no confidence, so every
    detection scores 1.0.
    """
    data = _payload(text)
    if not isinstance(data, list):
        return []
    by_lower = {t.lower(): t for t in targets}
    out: list[Detection] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        coords = item.get("bbox_2d")
        if not isinstance(label, str) or label.strip().lower() not in by_lower:
            continue
        if (
            not isinstance(coords, list)
            or len(coords) != 4
            or not all(isinstance(c, int | float) and not isinstance(c, bool) for c in coords)
        ):
            continue
        x1, y1, x2, y2 = (float(c) for c in coords)
        box = clip(
            Box(
                x1 / REL_SCALE * width,
                y1 / REL_SCALE * height,
                x2 / REL_SCALE * width,
                y2 / REL_SCALE * height,
            ),
            width,
            height,
        )
        if box.area <= 0:
            continue
        out.append(
            Detection(box=box, label=by_lower[label.strip().lower()], score=1.0, detector=detector)
        )
    return out
