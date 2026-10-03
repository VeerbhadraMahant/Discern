"""Baseline: the frozen v0 pipeline (classical restoration + YOLOv8n).

v0 chose its restorer with a proprietary VLM; here the restorer comes from the dataset-implied
scene label (an oracle, so this baseline is generous to v0). YOLOv8n only knows COCO classes,
so it scores zero recall on classes outside them (e.g. faces).
"""

import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import mlflow
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legacy" / "v0"))
import detection as v0_detection  # noqa: E402
import restoration as v0_restoration  # noqa: E402

from discern.config import load_settings  # noqa: E402
from discern.eval.datasets import load_dataset  # noqa: E402
from discern.eval.runner import cache_path, dataset_targets, detect_all, score_method  # noqa: E402
from discern.eval.types import AnnotatedImage  # noqa: E402
from discern.models.roles import Detection  # noqa: E402
from discern.vision.boxes import Box  # noqa: E402

NAME = "legacy-v0"
DATASETS = [
    "coco_val_clean",
    "bdd100k_clear",
    "bdd100k_rainy",
    "bdd100k_night",
    "hazydet_real",
    "darkface",
]
RESTORER_BY_SCENE = {"fog": "dehaze", "low_light": "low_light_enhancement"}


class V0Detector:
    name = NAME

    def detect(self, image: np.ndarray, targets: Sequence[str]) -> list[Detection]:
        bgr = np.ascontiguousarray(image[:, :, ::-1])
        return [
            Detection(box=Box(*d["xyxy"]), label=d["label"], score=d["confidence"], detector=NAME)
            for d in v0_detection.detect(bgr)
            if d["label"] in targets
        ]


def restore_by_scene(a: AnnotatedImage, rgb: np.ndarray) -> np.ndarray:
    name = RESTORER_BY_SCENE.get(a.scene_label or "")
    if name is None:
        return rgb
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return cv2.cvtColor(v0_restoration.RESTORATION_FUNCS[name](bgr), cv2.COLOR_BGR2RGB)


def main() -> None:
    settings = load_settings()
    mlruns = ROOT / "mlruns"
    mlruns.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(mlruns / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("m1-baselines")
    detector = V0Detector()
    for dataset in DATASETS:
        gate, harvest = load_dataset(dataset, "gate"), load_dataset(dataset, "harvest")
        targets = dataset_targets(gate + harvest)
        rev = "v0"
        gd = detect_all(
            detector, gate, targets, cache_path(NAME, f"{dataset}-gate", rev), restore_by_scene
        )
        hd = detect_all(
            detector,
            harvest,
            targets,
            cache_path(NAME, f"{dataset}-harvest", rev),
            restore_by_scene,
        )
        result = score_method(NAME, dataset, gate, gd, harvest, hd)
        with mlflow.start_run(run_name=f"{NAME}/{dataset}"):
            mlflow.log_params(
                {
                    "method": NAME,
                    "dataset": dataset,
                    "model_id": "yolov8n.pt + classical restoration",
                    "model_revision": rev,
                    "profile": settings.profile.name,
                    "config_hash": settings.config_hash,
                    "targets": ",".join(targets),
                    "threshold": result.threshold,
                    "gate_images": len(gate),
                }
            )
            c = result.counts
            mlflow.log_metrics(
                {
                    "f1": c.f1,
                    "precision": c.precision,
                    "recall": c.recall,
                    "tp": c.tp,
                    "fp": c.fp,
                    "fn": c.fn,
                }
            )
        print(f"{NAME:22s} {dataset:15s} F1={result.f1:.3f} thr={result.threshold:.2f}", flush=True)


if __name__ == "__main__":
    main()
