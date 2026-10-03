"""Milestone 3 ablations: single detectors vs multi-detector fusion vs full MED (adjudicated).

Input images are the SAIR output (`sair_full` from the Milestone 2 plans). Variants per dataset:
  single_<detector>  one detector, threshold tuned on the harvest split
  best_single        the single detector with the best harvest F1 (chosen without the gate)
  fused_k<K>         top-K detectors by harvest F1, per-detector operating thresholds, instance
                     grouping, anchors only (cheap fusion, no VLM)
  med_k2             the real pipeline: VLM detector_select (K=2) + grouping + adjudication of
                     the groups two detectors do not already agree on (gate split only)

Usage: uv run python scripts/run_ablations_m3.py [--limit N] [dataset ...]
Needs the Milestone 2 plan caches (run scripts/run_ablations_m2.py first).
"""

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ablations_m2 import (  # noqa: E402
    DEGRADED,
    MLRUNS,
    VLM_NAME,
    LazyRestorers,
    free_gpu,
    no_experience,
    plan_all,
    variant_images,
)

from discern.agent.med import detect_image  # noqa: E402
from discern.agent.schemas import DetectorInfo, SceneProfile, ShotPlan  # noqa: E402
from discern.config import load_settings  # noqa: E402
from discern.config.settings import Settings  # noqa: E402
from discern.eval.datasets import DATA_DIR, load_dataset  # noqa: E402
from discern.eval.metrics import Counts  # noqa: E402
from discern.eval.runner import (  # noqa: E402
    dataset_targets,
    detect_all,
    evaluate,
    load_rgb,
    tune_threshold,
)
from discern.eval.types import AnnotatedImage  # noqa: E402
from discern.models.loading import load_adapter  # noqa: E402
from discern.models.manager import ModelManager, RegistryEntry  # noqa: E402
from discern.models.registry import load_registry  # noqa: E402
from discern.models.roles import SCORE_FLOOR, Detection, Image  # noqa: E402
from discern.trace import TraceCollector  # noqa: E402
from discern.vision.grouping import group_detections  # noqa: E402

POOL = ["yolo-world-v2", "owlv2-base", "grounding-dino-base"]
CAPABILITIES = {
    "yolo-world-v2": "real-time open-vocabulary YOLO detector",
    "owlv2-base": "open-vocabulary detector, image-conditioned text queries",
    "grounding-dino-base": "text-grounded transformer detector",
}
DetMap = dict[str, list[Detection]]


def profile_from_plan(plan: ShotPlan) -> SceneProfile:
    """Rebuild the perceived profile from the key recorded in the plan's first decision."""
    scene, illum, vis, scale, density = plan.decisions[0].split(": ", 1)[1].split("|")
    return SceneProfile.model_validate(
        {
            "scene_label": scene,
            "illumination": illum,
            "visibility": vis,
            "object_scale": scale,
            "object_density": density,
            "confidence": 1.0,
        }
    )


class CachedDetector:
    """Serves precomputed raw detections for the image currently being processed."""

    def __init__(self, name: str, store: Mapping[str, Sequence[Detection]]) -> None:
        self.name = name
        self._store = store
        self.current = ""

    def detect(self, image: Image, targets: Sequence[str]) -> list[Detection]:
        return list(self._store[self.current])


def log_run(
    dataset: str,
    variant: str,
    counts: Counts,
    params: Mapping[str, str | int],
    extra: Mapping[str, float] | None = None,
) -> None:
    with mlflow.start_run(run_name=f"{variant}/{dataset}"):
        mlflow.log_params({"method": variant, "dataset": dataset, **params})
        mlflow.log_metrics(
            {
                "f1": counts.f1,
                "precision": counts.precision,
                "recall": counts.recall,
                **(extra or {}),
            }
        )
    print(f"{variant:22s} {dataset:15s} F1={counts.f1:.3f}", flush=True)


def subset(dets: Mapping[str, Sequence[Detection]], images: Sequence[AnnotatedImage]) -> DetMap:
    return {a.image_id: list(dets[a.image_id]) for a in images}


def fuse(
    images: Mapping[str, Image],
    raw: Mapping[str, DetMap],
    names: Sequence[str],
    op: Mapping[str, float],
    split: Sequence[AnnotatedImage],
    settings: Settings,
) -> DetMap:
    """Cheap fusion: pool detections above operating thresholds, group, keep the anchors."""
    out: DetMap = {}
    for a in split:
        pooled = [
            d for n in names for d in raw[n][a.image_id] if d.score >= max(op[n], SCORE_FLOOR)
        ]
        groups = group_detections(images[a.image_id], pooled, settings.thresholds.grouping)
        out[a.image_id] = [g.anchor for g in groups]
    return out


def run_dataset(
    dataset: str,
    manager: ModelManager,
    registry: Mapping[str, RegistryEntry],
    settings: Settings,
    limit: int,
    experience_for: Callable[[SceneProfile], str] = no_experience,
) -> None:
    """`experience_for` gives the detector-selection experience for a profile (default none)."""
    restorers = LazyRestorers(manager)
    catalog = [
        DetectorInfo(name=n, capabilities=CAPABILITIES[n], speed_class=registry[n].speed_class)
        for n in POOL
    ]

    def get_vlm() -> Any:
        return manager.get("agent_vlm", VLM_NAME)

    gate, harvest = load_dataset(dataset, "gate"), load_dataset(dataset, "harvest")
    if limit:
        gate, harvest = gate[:limit], harvest[:limit]
    images = gate + harvest
    targets = dataset_targets(images)
    tag = registry[VLM_NAME].revision[:10]
    plan_cache = DATA_DIR / "_cache" / "m2" / f"{dataset}-{tag}.json"
    plans = plan_all(images, get_vlm, restorers, plan_cache)
    sair = {
        a.image_id: variant_images(a, load_rgb(a.path), plans[a.image_id], restorers)["sair_full"]
        for a in images
    }

    raw: dict[str, DetMap] = {}
    for name in POOL:
        detector: Any = manager.get(registry[name].role, name)
        # The cached detections depend on the SAIR output (the VLM's plans) and the detector.
        key = f"{name}-{dataset}-{registry[name].revision[:10]}-{tag}"
        cache = DATA_DIR / "_cache" / "m3" / f"{key}.json"
        raw[name] = detect_all(
            detector, images, targets, cache, lambda a, _rgb, s=sair: s[a.image_id]
        )

    op = {n: tune_threshold(subset(raw[n], harvest), harvest) for n in POOL}
    harvest_f1 = {n: evaluate(subset(raw[n], harvest), harvest, op[n]).f1 for n in POOL}
    ranked = sorted(POOL, key=lambda n: -harvest_f1[n])
    params: dict[str, str | int] = {
        "pool": ",".join(POOL),
        "operating_thresholds": str(op),
        "config_hash": settings.config_hash,
        "gate_images": len(gate),
    }

    for n in POOL:
        log_run(dataset, f"single_{n}", evaluate(subset(raw[n], gate), gate, op[n]), params)
    best = ranked[0]
    log_run(dataset, "best_single", evaluate(subset(raw[best], gate), gate, op[best]), params)

    for k in range(1, len(POOL) + 1):
        fused = fuse(sair, raw, ranked[:k], op, gate, settings)
        log_run(dataset, f"fused_k{k}", evaluate(fused, gate, 0.0), params)

    cached = {n: CachedDetector(n, raw[n]) for n in POOL}
    trace = TraceCollector()
    med: DetMap = {}
    for a in gate:
        for c in cached.values():
            c.current = a.image_id
        profile = profile_from_plan(plans[a.image_id])
        med[a.image_id] = detect_image(
            get_vlm(),
            trace,
            sair[a.image_id],
            targets,
            profile,
            cached,  # type: ignore[arg-type]
            catalog,
            adjudicate_all=False,
            settings=settings,
            operating_thresholds=op,
            priority=ranked,
            experience=experience_for(profile),
        )
    extra = {
        "adjudicated_groups": float(sum(1 for e in trace.events if e.node == "adjudicate")),
        "fallback_events": float(sum(1 for e in trace.events if e.fallback_used)),
    }
    log_run(dataset, "med_k2", evaluate(med, gate, 0.0), params, extra)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DEGRADED)
    ap.add_argument("--limit", type=int, default=0, help="images per split (0 = all)")
    args = ap.parse_args()

    registry = load_registry()
    settings = load_settings()
    manager = ModelManager(
        registry, {}, settings.profile.vram_budget_gb, load_adapter, free_gpu
    )
    MLRUNS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(MLRUNS / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("m3-med")
    for dataset in args.datasets:
        run_dataset(dataset, manager, registry, settings, args.limit)


if __name__ == "__main__":
    main()
