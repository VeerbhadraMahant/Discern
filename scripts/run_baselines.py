"""Run single-detector baselines on every prepared dataset and log them to MLflow.

Usage: uv run python scripts/run_baselines.py [detector ...]   (default: all local detectors)
Thresholds are tuned on each dataset's harvest split and reported on its gate split.
"""

import sys
from functools import partial
from pathlib import Path

import mlflow

from discern.config import load_settings
from discern.eval.datasets import load_dataset
from discern.eval.runner import (
    EvalResult,
    cache_path,
    dataset_targets,
    detect_all,
    score_method,
)
from discern.models.loading import load_adapter
from discern.models.registry import load_registry

DATASETS = [
    "coco_val_clean",
    "bdd100k_clear",
    "bdd100k_rainy",
    "bdd100k_night",
    "hazydet_real",
    "darkface",
]
DETECTORS = ["yolo-world-v2", "owlv2-base", "grounding-dino-base"]
MLRUNS = Path(__file__).resolve().parents[1] / "mlruns"
TRACKING_URI = f"sqlite:///{(MLRUNS / 'mlflow.db').as_posix()}"


def _cache(name: str, dataset: str, revision: str, split: str) -> Path:
    return cache_path(name, f"{dataset}-{split}", revision)


def main(detectors: list[str]) -> list[EvalResult]:
    registry = load_registry()
    settings = load_settings()
    MLRUNS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("m1-baselines")
    results: list[EvalResult] = []
    for name in detectors:
        entry = registry[name]
        detector = load_adapter(entry)
        for dataset in DATASETS:
            gate, harvest = load_dataset(dataset, "gate"), load_dataset(dataset, "harvest")
            targets = dataset_targets(gate + harvest)
            if entry.role == "agent_vlm":  # constant score: skip the slow threshold-tuning pass
                harvest = []
            cache = partial(_cache, name, dataset, entry.revision)
            gate_dets = detect_all(detector, gate, targets, cache("gate"))
            harvest_dets = detect_all(detector, harvest, targets, cache("harvest"))
            result = score_method(name, dataset, gate, gate_dets, harvest, harvest_dets)
            with mlflow.start_run(run_name=f"{name}/{dataset}"):
                mlflow.log_params(
                    {
                        "method": name,
                        "dataset": dataset,
                        "model_id": entry.model_id,
                        "model_revision": entry.revision,
                        "profile": settings.profile.name,
                        "config_hash": settings.config_hash,
                        "targets": ",".join(targets),
                        "threshold": result.threshold,
                        "gate_images": len(gate),
                    }
                )
                mlflow.log_metrics(
                    {
                        "f1": result.counts.f1,
                        "precision": result.counts.precision,
                        "recall": result.counts.recall,
                        "tp": result.counts.tp,
                        "fp": result.counts.fp,
                        "fn": result.counts.fn,
                    }
                )
            results.append(result)
            print(
                f"{name:22s} {dataset:15s} F1={result.f1:.3f} thr={result.threshold:.2f}",
                flush=True,
            )
    return results


if __name__ == "__main__":
    main(sys.argv[1:] or DETECTORS)
