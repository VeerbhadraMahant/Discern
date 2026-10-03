"""Milestone 9 ablation: DetAS (no experience) versus DetAS-X (experience from a memory file).

Both arms run the same pipeline as the Milestone 3 `med_k2` variant on the gate split of each
degraded dataset: SAIR plans (experience at the restorer and SR nodes), the SAIR output as input
image, cached detections per detector, then `detect_image` (experience at the detector node).
  detas     no experience; reuses the Milestone 2 plan cache and the Milestone 3 detection caches
  detas_x   experience from data/memory/memory-<version>.json; its plans and detections are cached
            separately under data/_cache/m9/ (cache names carry the memory version), so the two
            arms and their caches never mix

Usage: uv run python scripts/run_ablation_m9.py --version v1 [--limit N] [dataset ...]
Needs the Milestone 2 plan caches and a memory built by scripts/harvest.py. Logs to the MLflow
experiment 'm9-detas-x'. Needs the VLM, so a GPU; never run by the tests.
"""

import argparse
import sys
from collections.abc import Callable, Mapping
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
from run_ablations_m3 import (  # noqa: E402
    CAPABILITIES,
    POOL,
    CachedDetector,
    DetMap,
    log_run,
    profile_from_plan,
    subset,
)

from discern.agent.med import detect_image  # noqa: E402
from discern.agent.schemas import DetectorInfo, SceneProfile  # noqa: E402
from discern.config import load_settings  # noqa: E402
from discern.config.settings import ExperienceThresholds  # noqa: E402
from discern.eval.datasets import DATA_DIR, load_dataset  # noqa: E402
from discern.eval.runner import (  # noqa: E402
    dataset_targets,
    detect_all,
    evaluate,
    load_rgb,
    tune_threshold,
)
from discern.experience.aggregate import Memory, load_memory, memory_path  # noqa: E402
from discern.experience.injection import render_for  # noqa: E402
from discern.experience.schema import Node  # noqa: E402
from discern.models.loading import load_adapter  # noqa: E402
from discern.models.manager import ModelManager  # noqa: E402
from discern.models.registry import load_registry  # noqa: E402
from discern.trace import TraceCollector  # noqa: E402

ExperienceFor = Callable[[SceneProfile], str]


def experience_for(memory: Memory, thresholds: ExperienceThresholds, *nodes: Node) -> ExperienceFor:
    def render(profile: SceneProfile) -> str:
        return render_for(memory, profile.key, nodes, thresholds)

    return render


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DEGRADED)
    ap.add_argument("--version", required=True, help="memory version id under data/memory/")
    ap.add_argument("--limit", type=int, default=0, help="images per split (0 = all)")
    args = ap.parse_args()

    registry = load_registry()
    settings = load_settings()
    memory = load_memory(memory_path(DATA_DIR / "memory", args.version))
    plan_experience = experience_for(memory, settings.thresholds.experience, "restorer", "sr")
    detect_experience = experience_for(memory, settings.thresholds.experience, "detector_set")
    manager = ModelManager(registry, {}, settings.profile.vram_budget_gb, load_adapter, free_gpu)
    restorers = LazyRestorers(manager)
    catalog = [
        DetectorInfo(name=n, capabilities=CAPABILITIES[n], speed_class=registry[n].speed_class)
        for n in POOL
    ]

    def get_vlm() -> Any:
        return manager.get("agent_vlm", VLM_NAME)

    MLRUNS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(MLRUNS / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("m9-detas-x")
    tag = registry[VLM_NAME].revision[:10]

    for dataset in args.datasets:
        gate, harvest = load_dataset(dataset, "gate"), load_dataset(dataset, "harvest")
        if args.limit:
            gate, harvest = gate[: args.limit], harvest[: args.limit]
        images = gate + harvest
        targets = dataset_targets(images)
        arms: Mapping[str, tuple[ExperienceFor, ExperienceFor, Path, Path]] = {
            "detas": (
                no_experience,
                no_experience,
                DATA_DIR / "_cache" / "m2" / f"{dataset}-{tag}.json",
                DATA_DIR / "_cache" / "m3",
            ),
            "detas_x": (
                plan_experience,
                detect_experience,
                DATA_DIR / "_cache" / "m9" / f"{dataset}-{tag}-{args.version}.json",
                DATA_DIR / "_cache" / "m9",
            ),
        }
        for arm, (plan_exp, detect_exp, plan_cache, det_cache_dir) in arms.items():
            plans = plan_all(images, get_vlm, restorers, plan_cache, plan_exp)
            sair = {
                a.image_id: variant_images(a, load_rgb(a.path), plans[a.image_id], restorers)[
                    "sair_full"
                ]
                for a in images
            }
            raw: dict[str, DetMap] = {}
            for name in POOL:
                detector: Any = manager.get(registry[name].role, name)
                key = f"{name}-{dataset}-{registry[name].revision[:10]}-{tag}"
                if arm == "detas_x":
                    key += f"-{args.version}"
                raw[name] = detect_all(
                    detector,
                    images,
                    targets,
                    det_cache_dir / f"{key}.json",
                    lambda a, _rgb, s=sair: s[a.image_id],
                )
            op = {n: tune_threshold(subset(raw[n], harvest), harvest) for n in POOL}
            ranked = sorted(
                POOL, key=lambda n: -evaluate(subset(raw[n], harvest), harvest, op[n]).f1
            )
            cached = {n: CachedDetector(n, raw[n]) for n in POOL}
            trace = TraceCollector()
            found: DetMap = {}
            for a in gate:
                for c in cached.values():
                    c.current = a.image_id
                profile = profile_from_plan(plans[a.image_id])
                found[a.image_id] = detect_image(
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
                    experience=detect_exp(profile),
                )
            params: dict[str, str | int] = {
                "pool": ",".join(POOL),
                "operating_thresholds": str(op),
                "config_hash": settings.config_hash,
                "gate_images": len(gate),
                "memory_version": args.version if arm == "detas_x" else "none",
            }
            log_run(dataset, arm, evaluate(found, gate, 0.0), params)


if __name__ == "__main__":
    main()
