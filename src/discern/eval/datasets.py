"""Dataset preparation and loading in the common annotation format.

Prepared datasets live under `data/<name>/` (gitignored): `annotations.json` is a list of
`AnnotatedImage` and image paths inside it are relative to `data/<name>/`.
"""

import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from discern.eval.types import AnnotatedImage, GroundTruthBox
from discern.vision.boxes import Box

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# BDD100K label -> the shared vocabulary detectors are prompted with. Others are dropped.
BDD_LABELS = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "pedestrian": "person",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
}

_ADAPTER = TypeAdapter(list[AnnotatedImage])


def bdd_sample_to_annotated(
    sample: Mapping[str, Any], scene_label: str, image_path: Path
) -> AnnotatedImage | None:
    """Convert one FiftyOne BDD100K sample. Returns None if no object in the vocabulary."""
    meta = sample["metadata"]
    w, h = meta["width"], meta["height"]
    objects = []
    for d in sample["detections"]["detections"]:
        label = BDD_LABELS.get(d["label"])
        if label is None:
            continue
        x, y, bw, bh = d["bounding_box"]  # relative xywh
        objects.append(
            GroundTruthBox(box=Box(x * w, y * h, (x + bw) * w, (y + bh) * h), label=label)
        )
    if not objects:
        return None
    return AnnotatedImage(
        image_id=Path(sample["filepath"]).stem,
        path=image_path,
        width=w,
        height=h,
        objects=tuple(objects),
        scene_label=scene_label,
    )


def split_subset(
    items: Sequence[AnnotatedImage], n_gate: int, n_harvest: int, seed: int
) -> list[AnnotatedImage]:
    """Seeded, disjoint gate and harvest subsets."""
    ordered = sorted(items, key=lambda a: a.image_id)
    random.Random(seed).shuffle(ordered)
    if len(ordered) < n_gate + n_harvest:
        raise ValueError(f"need {n_gate + n_harvest} images, have {len(ordered)}")
    gate = [a.model_copy(update={"split": "gate"}) for a in ordered[:n_gate]]
    harvest = [
        a.model_copy(update={"split": "harvest"}) for a in ordered[n_gate : n_gate + n_harvest]
    ]
    return gate + harvest


def save_annotations(name: str, items: Iterable[AnnotatedImage], data_dir: Path = DATA_DIR) -> Path:
    out = data_dir / name / "annotations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_ADAPTER.dump_json(list(items), indent=1))
    return out


def load_dataset(
    name: str, split: str | None = "gate", data_dir: Path = DATA_DIR
) -> list[AnnotatedImage]:
    """Load a prepared dataset; image paths are resolved to absolute paths."""
    root = data_dir / name
    items = _ADAPTER.validate_json((root / "annotations.json").read_bytes())
    return [
        a.model_copy(update={"path": root / a.path})
        for a in items
        if split is None or a.split == split
    ]
