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


def coco_to_annotated(
    coco: Mapping[str, Any],
    image_dir: Path,
    label_map: Mapping[str, str],
    scene_label: str,
    id_prefix: str = "",
) -> list[AnnotatedImage]:
    """Convert a COCO-format dict (absolute xywh boxes). Categories missing from
    `label_map` are dropped, and images left without objects are skipped."""
    names = {c["id"]: c["name"] for c in coco["categories"]}
    objects: dict[int, list[GroundTruthBox]] = {}
    for ann in coco["annotations"]:
        label = label_map.get(names[ann["category_id"]])
        if label is None:
            continue
        x, y, w, h = ann["bbox"]
        objects.setdefault(ann["image_id"], []).append(
            GroundTruthBox(box=Box(x, y, x + w, y + h), label=label)
        )
    return [
        AnnotatedImage(
            image_id=f"{id_prefix}{Path(img['file_name']).stem}",
            path=image_dir / Path(img["file_name"]).name,
            width=img["width"],
            height=img["height"],
            objects=tuple(objects[img["id"]]),
            scene_label=scene_label,
        )
        for img in coco["images"]
        if img["id"] in objects
    ]


def parse_darkface_label(text: str) -> tuple[GroundTruthBox, ...]:
    """DarkFace label file: a count line, then `x1 y1 x2 y2` (absolute pixels) per face."""
    lines = [ln.split() for ln in text.strip().splitlines()[1:] if ln.strip()]
    return tuple(
        GroundTruthBox(box=Box(*(float(v) for v in parts[:4])), label="face") for parts in lines
    )
