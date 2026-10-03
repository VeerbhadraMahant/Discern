from pathlib import Path

import pytest

from discern.eval.datasets import (
    bdd_sample_to_annotated,
    load_dataset,
    save_annotations,
    split_subset,
)
from discern.eval.types import AnnotatedImage, GroundTruthBox
from discern.vision.boxes import Box

SAMPLE = {
    "filepath": "data/abc-123.jpg",
    "metadata": {"width": 1000, "height": 500},
    "detections": {
        "detections": [
            {"label": "pedestrian", "bounding_box": [0.1, 0.2, 0.1, 0.4]},
            {"label": "traffic sign", "bounding_box": [0.5, 0.5, 0.1, 0.1]},
            {"label": "car", "bounding_box": [0.5, 0.5, 0.2, 0.2]},
        ]
    },
}


def test_bdd_conversion_scales_boxes_maps_labels_and_drops_others() -> None:
    a = bdd_sample_to_annotated(SAMPLE, "rain", Path("images/abc-123.jpg"))
    assert a is not None
    assert a.image_id == "abc-123" and a.scene_label == "rain"
    assert [o.label for o in a.objects] == ["person", "car"]
    assert a.objects[0].box == pytest.approx(Box(100, 100, 200, 300))


def test_bdd_conversion_skips_images_without_vocabulary_objects() -> None:
    only_signs = {**SAMPLE, "detections": {"detections": [SAMPLE["detections"]["detections"][1]]}}
    assert bdd_sample_to_annotated(only_signs, "rain", Path("x.jpg")) is None


def make(i: int) -> AnnotatedImage:
    return AnnotatedImage(
        image_id=f"img{i:03d}",
        path=Path(f"images/img{i:03d}.jpg"),
        width=10,
        height=10,
        objects=(GroundTruthBox(box=Box(0, 0, 1, 1), label="car"),),
    )


def test_split_subset_is_seeded_disjoint_and_sized() -> None:
    items = [make(i) for i in range(30)]
    a = split_subset(items, 10, 5, seed=0)
    assert a == split_subset(items, 10, 5, seed=0)
    assert a != split_subset(items, 10, 5, seed=1)
    gate = {x.image_id for x in a if x.split == "gate"}
    harvest = {x.image_id for x in a if x.split == "harvest"}
    assert len(gate) == 10 and len(harvest) == 5 and not gate & harvest
    with pytest.raises(ValueError):
        split_subset(items, 25, 10, seed=0)


def test_save_and_load_round_trip_resolves_paths(tmp_path: Path) -> None:
    items = split_subset([make(i) for i in range(6)], 4, 2, seed=0)
    save_annotations("toy", items, tmp_path)
    gate = load_dataset("toy", "gate", tmp_path)
    assert len(gate) == 4
    assert all(a.path.parent == tmp_path / "toy" / "images" for a in gate)
    assert len(load_dataset("toy", None, tmp_path)) == 6


def test_coco_conversion_maps_labels_and_converts_xywh() -> None:
    from discern.eval.datasets import coco_to_annotated

    coco = {
        "categories": [{"id": 0, "name": "car"}, {"id": 1, "name": "kite"}],
        "images": [
            {"id": 1, "file_name": "a/00001.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "00002.jpg", "width": 100, "height": 80},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 0, "bbox": [10, 20, 30, 40]},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]},
            {"image_id": 2, "category_id": 1, "bbox": [0, 0, 5, 5]},
        ],
    }
    (a,) = coco_to_annotated(coco, Path("images"), {"car": "car"}, "fog", id_prefix="hz_")
    assert a.image_id == "hz_00001" and a.path == Path("images/00001.jpg")
    assert a.objects == (GroundTruthBox(box=Box(10, 20, 40, 60), label="car"),)


def test_coco_conversion_skips_crowd_annotations() -> None:
    from discern.eval.datasets import coco_to_annotated

    coco = {
        "categories": [{"id": 0, "name": "car"}],
        "images": [
            {"id": 1, "file_name": "1.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "2.jpg", "width": 100, "height": 80},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 0, "bbox": [10, 20, 30, 40], "iscrowd": 0},
            {"image_id": 1, "category_id": 0, "bbox": [0, 0, 90, 70], "iscrowd": 1},
            {"image_id": 2, "category_id": 0, "bbox": [0, 0, 90, 70], "iscrowd": 1},
        ],
    }
    (a,) = coco_to_annotated(coco, Path("images"), {"car": "car"}, "normal")
    assert a.objects == (GroundTruthBox(box=Box(10, 20, 40, 60), label="car"),)


def test_darkface_label_parsing() -> None:
    from discern.eval.datasets import parse_darkface_label

    boxes = parse_darkface_label("2\r\n805 264 917 371\r\n558 338 584 374\r\n")
    assert [b.box for b in boxes] == [Box(805, 264, 917, 371), Box(558, 338, 584, 374)]
    assert all(b.label == "face" for b in boxes)
    assert parse_darkface_label("0\r\n") == ()
