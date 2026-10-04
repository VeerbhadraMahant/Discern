"""Milestone 9 ablation: DetAS (no experience), DetAS-X (experience text from the pinned memory)
DetAS-XP (node-wise experience-gated decision policy on top of the text) and DetAS-XJ (joint
configuration policy on top of the text).

All arms run the same pipeline on each degraded dataset and all APPLY the super-resolution their
plan decides (as the Engine does), so the only difference is the experience:
  detas     no experience anywhere; SAIR plans reuse the Milestone 2 plan cache
  detas_x   restorer and SR experience feed restorer_select, image_select and sr_select through
            plan_image; detector_set experience feeds detector_select (a VLM call only when the
            text is non-empty). Plans and detections are cached under data/_cache/m9/ with the
            memory version in their names.
  detas_xp  as detas_x, plus the ExperiencePolicy: a node whose retrieved evidence has at least
            experience.policy_min_count samples per compared option and a best-vs-runner-up F1
            margin of at least experience.policy_margin is decided by code with no VLM call (the
            VLM decides every other node). Its cache names also carry the policy settings.
Input image per arm: SAIR output (restored if the plan accepts it), then Real-ESRGAN by
plan.sr_factor.
Boxes are scaled back to the original frame and clipped. Per-detector operating thresholds are tuned
on the 50-image harvest split of the same arm; every metric is F1@0.5 (micro) on the 100-image gate:
  e2e_k2       PRIMARY. Per image detector_select picks K=2 detectors (deterministic priority order
               ranked by harvest F1 when the arm has no experience text), cheap fusion (pool above
               thresholds, group, anchors), no adjudication
  best_single  the single detector with the best harvest F1, for context
  e2e_k2_sub   e2e_k2 on the first --med-images gate images, comparable with med_k2
  med_k2       the full pipeline with VLM adjudication (adjudicate_all=False) on those images
Grouping and adjudication crops come from the original (unrestored) frame, as in the harvest.

  detas_xj  as detas_x, plus the JointExperiencePolicy: the best whole configuration (restorer, SR,
            detector set) from the memory's configuration stats, applied when it beats the default
            configuration (none, SR off, the first two of POOL) by experience.policy_margin with at
            least experience.policy_min_count samples. Needs configuration stats in the memory
            (scripts/add_config_stats.py). Its cache names carry the joint-policy settings.

Usage: uv run python scripts/run_ablation_m9.py [--limit N] [--med-images N]
           [--memory-version ID] [--arms detas,detas_x,detas_xp,detas_xj] [dataset ...]
An arm that is not run is read back from its latest MLflow run (same memory version, gate size and
med-images), so the final table still shows its deltas.
Needs the Milestone 2 plan caches and the pinned memory. Logs to the MLflow experiment 'm9-detas-x'.
Needs the VLM and detectors, so a GPU; never run by the tests.
"""

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ablations_m2 import (  # noqa: E402
    DEGRADED,
    MLRUNS,
    RESTORERS,
    VLM_NAME,
    LazyRestorers,
    free_gpu,
    no_experience,
    plan_all,
    predicted_scene,
)
from run_ablations_m3 import (  # noqa: E402
    CAPABILITIES,
    POOL,
    CachedDetector,
    DetMap,
    fuse,
    log_run,
    profile_from_plan,
    subset,
)

from discern.agent.med import detect_image  # noqa: E402
from discern.agent.nodes.detector_select import detector_select  # noqa: E402
from discern.agent.schemas import DetectorInfo, SceneProfile, ShotPlan  # noqa: E402
from discern.config import load_settings  # noqa: E402
from discern.config.settings import ExperienceThresholds  # noqa: E402
from discern.eval.ablation import (  # noqa: E402
    ARMS,
    NO_MEMORY,
    decision_stats,
    delta_table,
    detection_cache_name,
    pair_frequency,
    plan_cache_name,
    rescale_detections,
)
from discern.eval.datasets import DATA_DIR, load_dataset  # noqa: E402
from discern.eval.runner import (  # noqa: E402
    dataset_targets,
    detect_all,
    evaluate,
    load_rgb,
    tune_threshold,
)
from discern.eval.types import AnnotatedImage  # noqa: E402
from discern.experience.aggregate import Memory, load_memory, memory_path  # noqa: E402
from discern.experience.injection import render_for  # noqa: E402
from discern.experience.policy import (  # noqa: E402
    DecisionPolicy,
    ExperiencePolicy,
    JointExperiencePolicy,
    detector_decision,
    record_decision,
)
from discern.experience.promotion import read_pointer  # noqa: E402
from discern.experience.schema import Node  # noqa: E402
from discern.models.loading import load_adapter  # noqa: E402
from discern.models.manager import ModelManager  # noqa: E402
from discern.models.registry import load_registry  # noqa: E402
from discern.models.roles import Detection, Image  # noqa: E402
from discern.trace import TraceCollector  # noqa: E402

SR_NAME = "real-esrgan-x4plus"  # registry entry of the super_resolver role
METRICS = ["e2e_k2", "best_single", "e2e_k2_sub", "med_k2"]
ExperienceFor = Callable[[SceneProfile], str]
PolicyFor = Callable[[SceneProfile], DecisionPolicy | None]
EXPERIMENT = "m9-detas-x"
POLICY_ARM = "detas_xp"
JOINT_ARM = "detas_xj"


def experience_for(memory: Memory, thresholds: ExperienceThresholds, *nodes: Node) -> ExperienceFor:
    def render(profile: SceneProfile) -> str:
        return render_for(memory, profile.key, nodes, thresholds)

    return render


class LazyVlm:
    """Loads the VLM on the first call, so an arm that never asks it (no experience at the
    detector node) does not load it in the metric phase."""

    def __init__(self, get: Callable[[], Any]) -> None:
        self._get = get

    def generate(self, prompt: str, images: Sequence[Image] = ()) -> str:
        return str(self._get().generate(prompt, images))


class LazyDetector:
    """Loads its model through the ModelManager only when a detection is not cached."""

    def __init__(self, manager: ModelManager, role: str, name: str) -> None:
        self.name = name
        self._manager, self._role = manager, role

    def detect(self, image: Image, targets: Sequence[str]) -> list[Detection]:
        model: Any = self._manager.get(self._role, self.name)
        return list(model.detect(image, targets))


class ArmInputs:
    """Detector input per image: SAIR output (restored if the plan accepts it), then Real-ESRGAN
    by the plan's factor. Built once, on first use, and only when some detection is not cached."""

    def __init__(
        self,
        images: Sequence[AnnotatedImage],
        plans: Mapping[str, ShotPlan],
        restorers: LazyRestorers,
        manager: ModelManager,
    ) -> None:
        self._args = (images, plans, restorers, manager)
        self._built: dict[str, tuple[Image, int]] | None = None

    def _build(self) -> dict[str, tuple[Image, int]]:
        images, plans, restorers, manager = self._args
        built: dict[str, tuple[Image, int]] = {}
        for a in images:
            plan = plans[a.image_id]
            img = load_rgb(a.path)
            if plan.use_restored:
                img = restorers[plan.restorer].restore(img)
            factor = plan.sr_factor or 1
            if factor > 1:
                sr: Any = manager.get("super_resolver", SR_NAME)
                img = sr.upscale(img, factor)
            built[a.image_id] = (img, factor)
        manager.evict(SR_NAME)  # free VRAM for the detectors
        for _, entry in RESTORERS.values():
            manager.evict(entry)
        return built

    def _get(self) -> dict[str, tuple[Image, int]]:
        if self._built is None:
            self._built = self._build()
        return self._built

    def image(self, a: AnnotatedImage, _rgb: np.ndarray) -> np.ndarray:
        return self._get()[a.image_id][0]

    def back(self, a: AnnotatedImage, rgb: np.ndarray, dets: list[Detection]) -> list[Detection]:
        height, width = rgb.shape[:2]
        return rescale_detections(dets, self._get()[a.image_id][1], width, height)


def joint_rate(plans: Sequence[ShotPlan]) -> float:
    """Share of plans where the joint policy decided (its rationale is in plan.decisions)."""
    decided = sum(
        any(d.startswith("experience joint policy: ") for d in p.decisions) for p in plans
    )
    return decided / max(len(plans), 1)


def policy_rates(plans: Sequence[ShotPlan]) -> dict[str, float]:
    """Share of plans where the experience policy decided the restorer and the SR node."""
    n = max(len(plans), 1)
    return {
        f"policy_{node}_rate": sum(
            any(d.startswith(f"experience policy: {node} ") for d in p.decisions) for p in plans
        )
        / n
        for node in ("restorer", "sr")
    }


def prior_f1(
    arm: str, dataset: str, metrics: Sequence[str], params: Mapping[str, str | int]
) -> dict[tuple[str, str, str], float]:
    """Latest logged F1 of an arm that was not run now: same memory version and image counts."""
    wanted = " and ".join(f"params.{k} = '{v}'" for k, v in params.items())
    found: dict[tuple[str, str, str], float] = {}
    for metric in metrics:
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT],
            filter_string=f"attributes.run_name = '{arm}/{metric}/{dataset}' and {wanted}",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if len(runs):
            found[(dataset, arm, metric)] = float(runs["metrics.f1"].iloc[0])
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DEGRADED)
    ap.add_argument("--limit", type=int, default=0, help="images per split (0 = all)")
    ap.add_argument("--med-images", type=int, default=12, help="gate images for med_k2 (0 = all)")
    ap.add_argument("--memory-version", help="default: the version in the pinned pointer")
    ap.add_argument(
        "--arms", default=",".join(ARMS), help=f"comma list of arms to run, from {', '.join(ARMS)}"
    )
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms or any(a not in ARMS for a in arms):
        sys.exit(f"--arms takes a comma list from {', '.join(ARMS)}; got {args.arms!r}")

    registry = load_registry()
    settings = load_settings()
    memory_dir = DATA_DIR / "memory"
    pointer = memory_dir / settings.thresholds.serve.memory_pointer
    version = args.memory_version or read_pointer(pointer)
    if not version:
        sys.exit("no memory version: pin one (harvest.py --pin) or pass --memory-version")
    memory = load_memory(memory_path(memory_dir, version))
    exp = settings.thresholds.experience
    assert settings.thresholds.agent.top_k_detectors == 2, "metrics are named for K=2"
    plan_exp: dict[str, ExperienceFor] = {
        "detas": no_experience,
        "detas_x": experience_for(memory, exp, "restorer", "sr"),
        POLICY_ARM: experience_for(memory, exp, "restorer", "sr"),
        JOINT_ARM: experience_for(memory, exp, "restorer", "sr"),
    }
    det_exp: dict[str, ExperienceFor] = {
        "detas": no_experience,
        "detas_x": experience_for(memory, exp, "detector_set"),
        POLICY_ARM: experience_for(memory, exp, "detector_set"),
        JOINT_ARM: experience_for(memory, exp, "detector_set"),
    }
    k_detectors = settings.thresholds.agent.top_k_detectors
    if JOINT_ARM in arms and not memory.config_stats:
        sys.exit("detas_xj needs configuration stats: run scripts/add_config_stats.py first")

    def policy_of(profile: SceneProfile) -> ExperiencePolicy:
        return ExperiencePolicy.from_memory(memory, profile, exp)

    def joint_of(profile: SceneProfile) -> JointExperiencePolicy:
        # The default pair is the first K of the pool order: the harvest-ranked order needs
        # detections that depend on these plans.
        return JointExperiencePolicy.from_memory(memory, profile, exp, POOL, k_detectors)

    def no_policy(profile: SceneProfile) -> None:
        return None

    policy_for: dict[str, PolicyFor] = {
        "detas": no_policy,
        "detas_x": no_policy,
        POLICY_ARM: policy_of,
        JOINT_ARM: joint_of,
    }
    policy_key = f"{version}-pol{exp.policy_min_count}-{exp.policy_margin}"
    joint_key = f"{version}-joint{exp.policy_min_count}-{exp.policy_margin}-{'+'.join(POOL)}"
    memory_of = {
        "detas": NO_MEMORY,
        "detas_x": version,
        POLICY_ARM: policy_key,
        JOINT_ARM: joint_key,
    }

    manager = ModelManager(registry, {}, settings.profile.vram_budget_gb, load_adapter, free_gpu)
    restorers = LazyRestorers(manager)
    catalog = [
        DetectorInfo(name=n, capabilities=CAPABILITIES[n], speed_class=registry[n].speed_class)
        for n in POOL
    ]

    def get_vlm() -> Any:
        return manager.get("agent_vlm", VLM_NAME)

    vlm = LazyVlm(get_vlm)
    MLRUNS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(MLRUNS / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT)
    tag = registry[VLM_NAME].revision[:10]
    results: dict[tuple[str, str, str], float] = {}

    for dataset in args.datasets:
        gate, harvest = load_dataset(dataset, "gate"), load_dataset(dataset, "harvest")
        if args.limit:
            gate, harvest = gate[: args.limit], harvest[: args.limit]
        images = gate + harvest
        targets = dataset_targets(images)
        sub = gate[: args.med_images] if args.med_images else gate

        # Phase 1: SAIR plans for both arms (VLM), then free the VLM.
        plan_files = {
            "detas": DATA_DIR / "_cache" / "m2" / f"{dataset}-{tag}.json",
            "detas_x": DATA_DIR / "_cache" / "m9" / plan_cache_name(dataset, tag, version),
            POLICY_ARM: DATA_DIR
            / "_cache"
            / "m9"
            / plan_cache_name(dataset, tag, memory_of[POLICY_ARM], POLICY_ARM),
            JOINT_ARM: DATA_DIR
            / "_cache"
            / "m9"
            / plan_cache_name(dataset, tag, memory_of[JOINT_ARM], JOINT_ARM),
        }
        plans = {
            arm: plan_all(
                images, get_vlm, restorers, plan_files[arm], plan_exp[arm], policy_for[arm]
            )
            for arm in arms
        }
        manager.evict(VLM_NAME)

        # Phase 2: restore, super-resolve, detect, scale back; cached per arm and detector.
        raw: dict[str, dict[str, DetMap]] = {}
        for arm in arms:
            inputs = ArmInputs(images, plans[arm], restorers, manager)
            raw[arm] = {}
            for name in POOL:
                cache = DATA_DIR / "_cache" / "m9" / detection_cache_name(
                    arm, name, dataset, registry[name].revision, tag, memory_of[arm]
                )
                raw[arm][name] = detect_all(
                    LazyDetector(manager, registry[name].role, name),  # type: ignore[arg-type]
                    images,
                    targets,
                    cache,
                    preprocess=inputs.image,
                    postprocess=inputs.back,
                )
                manager.evict(name)

        # Phase 3: metrics on the gate split (the VLM is reloaded on demand).
        for arm in arms:
            op = {n: tune_threshold(subset(raw[arm][n], harvest), harvest) for n in POOL}
            harvest_f1 = {
                n: evaluate(subset(raw[arm][n], harvest), harvest, op[n]).f1 for n in POOL
            }
            ranked = sorted(POOL, key=lambda n: -harvest_f1[n])
            gate_plans = [plans[arm][a.image_id] for a in gate]
            stats = decision_stats(
                gate_plans,
                [predicted_scene(p) for p in gate_plans],
                [a.scene_label for a in gate],
            )

            trace = TraceCollector()
            fused: DetMap = {}
            chosen: list[list[str]] = []
            with_experience = 0
            by_policy = 0
            for a in gate:
                profile = profile_from_plan(plans[arm][a.image_id])
                text = det_exp[arm](profile)
                with_experience += bool(text.strip())
                decision = detector_decision(policy_for[arm](profile), k_detectors, POOL)
                if decision is not None:
                    record_decision(trace, decision)
                    by_policy += 1
                choice = detector_select(
                    vlm,
                    trace,
                    targets,
                    profile,
                    catalog,
                    ranked,
                    text,
                    settings,
                    None if decision is None else decision.detectors,
                )
                chosen.append(choice.detectors)
                frame = {a.image_id: load_rgb(a.path)}
                fused.update(fuse(frame, raw[arm], choice.detectors, op, [a], settings))
            stats["detector_experience_rate"] = with_experience / len(gate)
            if arm == POLICY_ARM:
                stats.update(policy_rates(gate_plans))
                stats["policy_detector_rate"] = by_policy / len(gate)
            if arm == JOINT_ARM:
                stats["policy_joint_rate"] = joint_rate(gate_plans)
                stats["policy_detector_rate"] = by_policy / len(gate)

            params: dict[str, str | int] = {
                "arm": arm,
                "memory_version": memory_of[arm],
                "pool": ",".join(POOL),
                "operating_thresholds": str(op),
                "config_hash": settings.config_hash,
                "gate_images": len(gate),
                "med_images": len(sub),
                "detector_pairs": str(pair_frequency(chosen)),
            }
            best = ranked[0]
            scored = {
                "e2e_k2": evaluate(fused, gate, 0.0),
                "best_single": evaluate(subset(raw[arm][best], gate), gate, op[best]),
                "e2e_k2_sub": evaluate(subset(fused, sub), sub, 0.0),
            }

            cached = {n: CachedDetector(n, raw[arm][n]) for n in POOL}
            med_trace = TraceCollector()
            med: DetMap = {}
            for a in sub:
                for c in cached.values():
                    c.current = a.image_id
                profile = profile_from_plan(plans[arm][a.image_id])
                decision = detector_decision(policy_for[arm](profile), k_detectors, POOL)
                med[a.image_id] = detect_image(
                    vlm,
                    med_trace,
                    load_rgb(a.path),
                    targets,
                    profile,
                    cached,  # type: ignore[arg-type]
                    catalog,
                    adjudicate_all=False,
                    settings=settings,
                    operating_thresholds=op,
                    priority=ranked,
                    experience=det_exp[arm](profile),
                    preferred=None if decision is None else decision.detectors,
                )
            scored["med_k2"] = evaluate(med, sub, 0.0)
            extra = {
                "adjudicated_groups": float(
                    sum(1 for e in med_trace.events if e.node == "adjudicate")
                ),
                "fallback_events": float(sum(1 for e in med_trace.events if e.fallback_used)),
            }

            for metric, counts in scored.items():
                run_extra = extra if metric == "med_k2" else None
                log_run(dataset, f"{arm}/{metric}", counts, params, run_extra)
                results[(dataset, arm, metric)] = counts.f1
            with mlflow.start_run(run_name=f"{arm}/decisions/{dataset}"):
                mlflow.log_params({"arm": arm, "dataset": dataset, **params})
                mlflow.log_metrics(stats)
            print(f"{arm}/decisions {dataset}: {stats} pairs={pair_frequency(chosen)}", flush=True)

        for arm in ARMS:  # arms not run now come from their latest logged run
            if arm not in arms:
                logged = {
                    "memory_version": memory_of[arm],
                    "gate_images": len(gate),
                    "med_images": len(sub),
                }
                results.update(prior_f1(arm, dataset, METRICS, logged))

    print()
    print(delta_table(results, args.datasets, METRICS))


if __name__ == "__main__":
    main()
