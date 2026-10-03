"""Offline SEEH harvest: build an experience memory from each dataset's harvest split.

Usage:
  uv run python scripts/harvest.py --version v1 --restorers dehaze=<registry entry> \
      --sr <registry entry> --detectors yolo-world-v2 owlv2-base grounding-dino-base \
      --no-confirm [--limit N] [dataset ...]

Inputs: data/<name> prepared datasets (harvest split, `discern.eval.datasets.load_dataset`).
Detector outputs are cached under data/_cache/seeh/, one file per (variant, detector, dataset).
Outputs: data/memory/records.jsonl (raw records, appended) and
data/memory/memory-<version>.json (aggregated, versioned memory).

Full-adjudication confirmation of the top configurations is not wired yet (it needs the GPU
adjudication pass); `--no-confirm` accepts the cheap fused score for them and says so. Records
written that way carry no adjudicated rows, so do not report them as paper-faithful.
Publishing to Hugging Face is a separate step: `discern.experience.aggregate.publish_to_hf`.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("datasets", nargs="*", help="dataset names (default: the six benchmark sets)")
    p.add_argument("--version", required=True, help="memory version id, for example v1")
    p.add_argument("--vlm", default="qwen3-vl-4b-4bit", help="registry entry used for perception")
    p.add_argument("--detectors", nargs="+", required=True, help="registry entries (the pool)")
    p.add_argument(
        "--restorers",
        nargs="*",
        default=[],
        metavar="NAME=ENTRY",
        help="restorer role name -> registry entry, for example dehaze=ridcp",
    )
    p.add_argument("--sr", default=None, help="registry entry of the super-resolution model")
    p.add_argument("--limit", type=int, default=None, help="images per dataset (default: settings)")
    p.add_argument("--min-score", type=float, default=0.0, help="detection score floor")
    p.add_argument("--no-confirm", action="store_true", help="skip full adjudication (see above)")
    p.add_argument("--memory-dir", type=Path, default=None, help="default: data/memory")
    return p.parse_args(argv)


DATASETS = [
    "coco_val_clean",
    "bdd100k_clear",
    "bdd100k_rainy",
    "bdd100k_night",
    "hazydet_real",
    "darkface",
]


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)
    # Heavy imports live here so `--help` and linting never load model stacks.
    import numpy as np

    from discern.agent.nodes.perception import perception
    from discern.agent.nodes.restorer_select import RESTORER_FOR_SCENE
    from discern.agent.nodes.sr_select import required_factor
    from discern.config import load_settings
    from discern.eval.datasets import DATA_DIR, load_dataset
    from discern.eval.runner import cache_path, dataset_targets, detect_all, load_rgb
    from discern.eval.types import GroundTruthBox
    from discern.experience.aggregate import build_memory, write_memory
    from discern.experience.harvest import (
        NONE,
        SR_AUTO,
        SR_OFF,
        CachedOutputs,
        Configuration,
        fused_f1,
        harvest_image,
    )
    from discern.experience.schema import ExperienceStore
    from discern.models.loading import load_adapter
    from discern.models.registry import load_registry
    from discern.models.roles import Image
    from discern.trace import TraceCollector
    from discern.vision.boxes import Box

    settings = load_settings()
    exp = settings.thresholds.experience
    registry = load_registry()
    memory_dir = args.memory_dir or DATA_DIR / "memory"
    store = ExperienceStore(memory_dir / "records.jsonl")

    vlm = load_adapter(registry[args.vlm])
    detectors = {name: load_adapter(registry[name]) for name in args.detectors}
    restorers = {
        name: load_adapter(registry[entry])
        for name, entry in (r.split("=", 1) for r in args.restorers)
    }
    sr_model = load_adapter(registry[args.sr]) if args.sr else None
    if not args.no_confirm:
        raise SystemExit("full-adjudication confirmation is not wired yet; pass --no-confirm")
    print("WARNING: confirmation skipped, cheap fused scores stand in for adjudication")

    grouping = settings.thresholds.grouping
    target = settings.thresholds.agent.sr_target_long_side
    limit = args.limit or exp.harvest_samples_per_dataset
    records = []
    for dataset in args.datasets or DATASETS:
        images = load_dataset(dataset, "harvest")[:limit]
        targets = dataset_targets(images)
        rgb = {a.image_id: load_rgb(a.path) for a in images}
        profiles = {a.image_id: perception(vlm, TraceCollector(), rgb[a.image_id]) for a in images}
        outputs: dict[str, CachedOutputs] = {a.image_id: {} for a in images}  # type: ignore[assignment]
        for restorer in [NONE, *restorers]:
            for sr in (SR_OFF, SR_AUTO):
                if sr == SR_AUTO and sr_model is None:
                    continue

                def variant(
                    a: object, img: np.ndarray, r: str = restorer, s: str = sr
                ) -> np.ndarray:
                    out = img if r == NONE else restorers[r].restore(img)
                    factor = required_factor(max(out.shape[:2]), target)
                    if s == SR_AUTO and factor and sr_model is not None:
                        out = sr_model.upscale(out, factor)
                    return out

                for name, detector in detectors.items():
                    cache = cache_path(f"seeh-{restorer}-{sr}-{name}", f"{dataset}-harvest", "v1")
                    found = detect_all(detector, images, targets, cache, preprocess=variant)
                    for a in images:
                        factor = 1
                        if sr == SR_AUTO:  # boxes come back in upscaled pixels: map to original
                            long_side = max(rgb[a.image_id].shape[:2])
                            factor = required_factor(long_side, target) or 1
                        scaled = [
                            d.model_copy(update={"box": Box(*(c / factor for c in d.box))})
                            for d in found[a.image_id]
                        ]
                        outputs[a.image_id].setdefault((restorer, sr), {})[name] = scaled
        for a in images:
            mapped = RESTORER_FOR_SCENE.get(profiles[a.image_id].scene_label, NONE)
            if mapped not in restorers:
                mapped = NONE
            image, gt, outs = rgb[a.image_id], a.objects, outputs[a.image_id]

            def confirm(
                c: Configuration,
                image: Image = image,
                gt: Sequence[GroundTruthBox] = gt,
                outs: CachedOutputs = outs,
            ) -> float:
                return fused_f1(c, image, gt, outs, grouping, args.min_score)

            records += harvest_image(
                a.image_id,
                profiles[a.image_id],
                "detect",
                image,
                gt,
                outs,
                mapped,
                list(detectors),
                grouping,
                confirm,
                0 if args.no_confirm else exp.confirm_top_configs,  # 0: no "adjudicated" rows
                args.version,
                args.min_score,
            )
            print(f"{dataset} {a.image_id}: {len(records)} records", flush=True)

    store.append(records)
    memory = build_memory(store.load(), args.version)
    print("wrote", write_memory(memory, memory_dir), f"({memory.version.record_count} records)")


if __name__ == "__main__":
    main()
