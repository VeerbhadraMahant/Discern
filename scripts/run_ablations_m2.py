"""Milestone 2 ablations on the degraded datasets (SR is decided but not applied here).

Variants, all scored with the same detector and a threshold tuned on the harvest split:
  none            original image
  always          oracle restorer for the dataset-implied scene label, always applied
  sair_no_select  perception + restorer_select, restored image used without image_select
  sair_full       perception + restorer_select + image_select (original or restored)

Usage: uv run python scripts/run_ablations_m2.py [--stub-vlm] [--limit N] [dataset ...]
--stub-vlm answers every VLM call with invalid JSON, so every node takes its deterministic
fallback (a plumbing check and a "no VLM" ablation, not a real result).
"""

import argparse
import gc
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from discern.agent.nodes.restorer_select import NONE, RESTORER_FOR_SCENE
from discern.agent.sair import plan_image
from discern.agent.schemas import ShotPlan
from discern.config import load_settings
from discern.eval.datasets import DATA_DIR, load_dataset
from discern.eval.runner import dataset_targets, load_rgb, score_method
from discern.eval.types import AnnotatedImage
from discern.models.loading import load_adapter
from discern.models.manager import ModelManager, RegistryEntry
from discern.models.registry import load_registry
from discern.models.roles import VLM, Detection, Image, Restorer
from discern.trace import TraceCollector

VLM_NAME, DETECTOR_NAME = "qwen3-vl-4b-4bit", "owlv2-base"
DEGRADED = ["hazydet_real", "bdd100k_night", "bdd100k_rainy", "darkface"]
VARIANTS = ["none", "always", "sair_no_select", "sair_full"]
# plan_image restorer name -> (registry role, registry entry). RIDCP has no downloadable weights,
# so dehaze uses the classical dark-channel-prior fallback.
RESTORERS = {
    "dehaze": ("restorer_dehaze", "classical-dehaze"),
    "derain": ("restorer_derain", "mprnet-derain"),
    "denoise": ("restorer_denoise", "swinir-denoise"),
    "lowlight": ("restorer_lowlight", "zero-dce-pp"),
}
MLRUNS = Path(__file__).resolve().parents[1] / "mlruns"


class StubVLM:
    def generate(self, prompt: str, images: Sequence[Image] = ()) -> str:
        return "{}"


class LazyRestorers(Mapping[str, Restorer]):
    """Restorers loaded on demand through the ModelManager (LRU eviction keeps VRAM bounded)."""

    def __init__(self, manager: ModelManager) -> None:
        self._manager = manager

    def __getitem__(self, name: str) -> Restorer:
        role, entry = RESTORERS[name]
        restorer: Any = self._manager.get(role, entry)
        return restorer  # type: ignore[no-any-return]

    def __contains__(self, name: object) -> bool:
        return name in RESTORERS  # the default Mapping check would load the model to answer

    def __iter__(self) -> Iterator[str]:
        return iter(RESTORERS)

    def __len__(self) -> int:
        return len(RESTORERS)


def free_gpu(model: object) -> None:
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def _save(path: Path, plans: Mapping[str, ShotPlan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v.model_dump() for k, v in plans.items()}))


def plan_all(
    images: Sequence[AnnotatedImage],
    get_vlm: Callable[[], VLM],
    restorers: LazyRestorers,
    cache_file: Path,
) -> dict[str, ShotPlan]:
    """SAIR plan per image, cached on disk so an interrupted run resumes."""
    plans: dict[str, ShotPlan] = {}
    if cache_file.exists():
        raw = json.loads(cache_file.read_text())
        plans = {k: ShotPlan.model_validate(v) for k, v in raw.items()}
    trace = TraceCollector()
    for n, a in enumerate(images):
        if a.image_id in plans:
            continue
        plans[a.image_id], _ = plan_image(get_vlm(), trace, load_rgb(a.path), restorers)
        if n % 10 == 9:
            _save(cache_file, plans)
    _save(cache_file, plans)
    return plans


def predicted_scene(plan: ShotPlan) -> str:
    return plan.decisions[0].split(": ", 1)[1].split("|", 1)[0]


def variant_images(
    a: AnnotatedImage, rgb: Image, plan: ShotPlan, restorers: LazyRestorers
) -> dict[str, Image]:
    cache: dict[str, Image] = {}

    def restored(name: str) -> Image:
        if name not in cache:
            cache[name] = restorers[name].restore(rgb)
        return cache[name]

    oracle = {"fog": "dehaze", "rain": "derain", "low_light": "lowlight", "noise": "denoise"}.get(
        a.scene_label or "", NONE
    )
    assert RESTORER_FOR_SCENE["fog"] == "dehaze"  # keep the oracle map in sync with the node
    return {
        "none": rgb,
        "always": rgb if oracle == NONE else restored(oracle),
        "sair_no_select": rgb if plan.restorer == NONE else restored(plan.restorer),
        "sair_full": restored(plan.restorer) if plan.use_restored else rgb,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DEGRADED)
    ap.add_argument("--stub-vlm", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="images per split (0 = all)")
    args = ap.parse_args()

    registry: dict[str, RegistryEntry] = load_registry()
    settings = load_settings()
    manager = ModelManager(
        registry, {}, settings.profile.vram_budget_gb, load_adapter, free_gpu
    )
    restorers = LazyRestorers(manager)
    stub = StubVLM()

    def get_vlm() -> VLM:
        vlm: Any = stub if args.stub_vlm else manager.get("agent_vlm", VLM_NAME)
        return vlm  # type: ignore[no-any-return]

    MLRUNS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(MLRUNS / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("m2-ablations-stub" if args.stub_vlm else "m2-ablations")

    for dataset in args.datasets:
        gate, harvest = load_dataset(dataset, "gate"), load_dataset(dataset, "harvest")
        if args.limit:
            gate, harvest = gate[: args.limit], harvest[: args.limit]
        targets = dataset_targets(gate + harvest)
        tag = "stub" if args.stub_vlm else registry[VLM_NAME].revision[:10]
        cache_file = DATA_DIR / "_cache" / "m2" / f"{dataset}-{tag}.json"
        plans = plan_all(gate + harvest, get_vlm, restorers, cache_file)

        dets: dict[str, dict[str, list[Detection]]] = {v: {} for v in VARIANTS}
        for a in gate + harvest:
            rgb = load_rgb(a.path)
            for variant, img in variant_images(a, rgb, plans[a.image_id], restorers).items():
                detector: Any = manager.get("detector_accurate", DETECTOR_NAME)
                dets[variant][a.image_id] = detector.detect(img, targets)

        gate_plans = [plans[a.image_id] for a in gate]
        pairs = zip(gate_plans, gate, strict=True)
        acc = float(np.mean([predicted_scene(p) == a.scene_label for p, a in pairs]))
        restore_rate = float(np.mean([p.restorer != NONE for p in gate_plans]))
        accept_rate = float(np.mean([p.use_restored for p in gate_plans]))
        for variant in VARIANTS:
            r = score_method(
                variant,
                dataset,
                gate,
                {a.image_id: dets[variant][a.image_id] for a in gate},
                harvest,
                {a.image_id: dets[variant][a.image_id] for a in harvest},
            )
            with mlflow.start_run(run_name=f"{variant}/{dataset}"):
                mlflow.log_params(
                    {
                        "method": variant,
                        "dataset": dataset,
                        "detector": DETECTOR_NAME,
                        "vlm": "stub" if args.stub_vlm else VLM_NAME,
                        "threshold": r.threshold,
                        "config_hash": settings.config_hash,
                        "gate_images": len(gate),
                    }
                )
                mlflow.log_metrics(
                    {
                        "f1": r.f1,
                        "precision": r.counts.precision,
                        "recall": r.counts.recall,
                        "perception_acc": acc,
                        "restore_rate": restore_rate,
                        "accept_rate": accept_rate,
                    }
                )
            print(
                f"{variant:15s} {dataset:15s} F1={r.f1:.3f} thr={r.threshold:.2f} "
                f"percep_acc={acc:.2f} restore={restore_rate:.2f} accept={accept_rate:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
