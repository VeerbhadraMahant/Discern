"""Diagnose the adjudicate node: how well does the VLM separate real objects from false groups?

For the first N gate images of a dataset, groups the cached detections of the top-2 detectors,
adjudicates every group the detectors do not already agree on, and compares the VLM's decision
with ground truth (does the anchor box match a ground-truth box of the same label at IoU 0.5?).
Needs the Milestone 3 detection caches. Usage: diagnose_adjudication.py [dataset] [n_images]
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ablations_m2 import (  # noqa: E402  # noqa: E402
    VLM_NAME,
    LazyRestorers,
    free_gpu,
    plan_all,
    variant_images,
)
from run_ablations_m3 import POOL, DetMap  # noqa: E402

from discern.agent.nodes.adjudicate import adjudicate  # noqa: E402
from discern.config import load_settings  # noqa: E402
from discern.eval.datasets import DATA_DIR, load_dataset  # noqa: E402
from discern.eval.runner import (  # noqa: E402
    dataset_targets,
    detect_all,
    evaluate,
    load_rgb,
    tune_threshold,
)
from discern.models.loading import load_adapter  # noqa: E402
from discern.models.manager import ModelManager  # noqa: E402
from discern.models.registry import load_registry  # noqa: E402
from discern.trace import TraceCollector  # noqa: E402
from discern.vision.boxes import iou  # noqa: E402
from discern.vision.grouping import group_detections  # noqa: E402


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "bdd100k_night"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    registry, settings = load_registry(), load_settings()
    manager = ModelManager(registry, {}, settings.profile.vram_budget_gb, load_adapter, free_gpu)
    gate = load_dataset(dataset, "gate")[:n]
    harvest = load_dataset(dataset, "harvest")
    tag = registry[VLM_NAME].revision[:10]
    images = load_dataset(dataset, "gate") + harvest
    targets = dataset_targets(images)
    restorers = LazyRestorers(manager)

    def get_vlm() -> Any:
        return manager.get("agent_vlm", VLM_NAME)

    plans = plan_all(
        images, get_vlm, restorers, DATA_DIR / "_cache" / "m2" / f"{dataset}-{tag}.json"
    )
    sair = {
        a.image_id: variant_images(a, load_rgb(a.path), plans[a.image_id], restorers)["sair_full"]
        for a in images
    }
    raw: dict[str, DetMap] = {}
    for name in POOL:
        key = f"{name}-{dataset}-{registry[name].revision[:10]}-{tag}"
        raw[name] = detect_all(None, images, targets, DATA_DIR / "_cache" / "m3" / f"{key}.json")  # type: ignore[arg-type]
    op = {
        nm: tune_threshold({a.image_id: raw[nm][a.image_id] for a in harvest}, harvest)
        for nm in POOL
    }
    harvest_f1 = {
        nm: evaluate({a.image_id: raw[nm][a.image_id] for a in harvest}, harvest, op[nm]).f1
        for nm in POOL
    }
    ranked = sorted(POOL, key=lambda nm: -harvest_f1[nm])[:2]  # by harvest F1, as in m3

    stats: Counter[str] = Counter()
    trace = TraceCollector()
    for a in gate:
        pooled = [d for nm in ranked for d in raw[nm][a.image_id] if d.score >= op[nm]]
        for g in group_detections(sair[a.image_id], pooled, settings.thresholds.grouping):
            if len({m.detector for m in g.members if m.label == g.anchor.label}) >= 2:
                continue  # detectors agree: accepted without the VLM
            real = any(
                o.label == g.anchor.label and iou(o.box, g.anchor.box) >= 0.5 for o in a.objects
            )
            result = adjudicate(get_vlm(), trace, sair[a.image_id], g, targets, settings)
            kind = "real" if real else "false"
            stats[f"{kind}:{'rejected' if result is None else 'accepted'}"] += 1
    print(dict(stats))
    tp, fp = stats["real:accepted"], stats["false:accepted"]
    real_total = tp + stats["real:rejected"]
    false_total = fp + stats["false:rejected"]
    print(f"real groups kept {tp}/{real_total}, false groups kept {fp}/{false_total}")


if __name__ == "__main__":
    main()
