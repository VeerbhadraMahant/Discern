"""Offline SEEH harvest, step 2: build a versioned experience memory from cached detections.

Usage (after scripts/harvest_variants.py has filled data/_cache/seeh/):
  uv run python scripts/harvest.py [--pin] [--limit N] [--adjudicate] [dataset ...]
Default datasets: hazydet_real bdd100k_night bdd100k_rainy darkface.

Inputs: each dataset's harvest split (`discern.eval.datasets.load_dataset`), its cached detections
(format in `discern.experience.harvest_cache`) and ground truth. The scene profile of an image is
the dataset-implied scene label plus the statistics-based attributes of the original image
(`discern.vision.stats.profile_from_stats`), so no VLM is needed; the mapped restorer comes from
the same label.
Outputs under data/memory/ (or --memory-dir): records-<version>.jsonl (raw records of this run)
and memory-<version>.json (aggregated). The version id is <date>-<record count> unless --version
is given. The pinned pointer file (what the app reads) is written only with --pin.

Confirmation of the top configurations: with `--adjudicate` each is re-scored by running
`detect_image` (VLM detector selection among the configuration's detectors, then crop-level
adjudication of every group) on the cached detections; this needs the VLM and so a GPU. Without
it (the default; `--no-confirm` is accepted and means the same) the cheap fused score stands in
and the records carry no adjudicated rows, so do not report them as paper-faithful. Not wired:
the VLM may pick fewer detectors than the configuration names, because selection is not forced.
Publishing to Hugging Face is a separate step: `discern.experience.aggregate.publish_to_hf`.
"""

import argparse
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from discern.experience.aggregate import OptionStat

DATASETS = ["hazydet_real", "bdd100k_night", "bdd100k_rainy", "darkface"]
NODE_ORDER = {"restorer": 0, "sr": 1, "detector_set": 2}


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("datasets", nargs="*", default=DATASETS, help="dataset names")
    p.add_argument("--version", default=None, help="memory version id (default: <date>-<records>)")
    p.add_argument("--vlm", default="qwen3-vl-4b-4bit", help="registry entry for --adjudicate")
    p.add_argument("--detectors", nargs="+", default=None, help="the pool (default: harvest pool)")
    p.add_argument("--limit", type=int, default=None, help="images per dataset (default: settings)")
    p.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="one floor for all detectors (default: tune one per detector)",
    )
    p.add_argument("--adjudicate", action="store_true", help="confirm top configs (see above)")
    p.add_argument("--no-confirm", action="store_true", help="skip adjudication (the default)")
    p.add_argument("--pin", action="store_true", help="write the pinned pointer to this version")
    p.add_argument("--memory-dir", type=Path, default=None, help="default: data/memory")
    p.add_argument("--cache-dir", type=Path, default=None, help="default: data/_cache/seeh")
    return p.parse_args(argv)


class _Cached:
    """A detector that returns precomputed detections whatever the image."""

    def __init__(self, detections: Sequence[object]) -> None:
        self._detections = list(detections)

    def detect(self, image: object, targets: Sequence[str]) -> list[object]:
        return list(self._detections)


def summary_rows(stats: Iterable[OptionStat]) -> list[tuple[str, str, str, float, int]]:
    """(scene label, node, option, mean node value, image count): profile keys of the same scene
    label pooled, weighted by their image counts."""
    total: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for s in stats:
        acc = total[(s.profile_key.split("|")[0], s.node, s.option)]
        acc[0] += s.mean * s.count
        acc[1] += s.count
    rows = [(*k, v[0] / v[1], int(v[1])) for k, v in total.items()]
    return sorted(rows, key=lambda r: (r[0], NODE_ORDER[r[1]], -r[3]))


def print_summary(stats: Iterable[OptionStat]) -> None:
    print(f"{'scene':10s} {'node':13s} {'option':44s} {'mean F1':>8s} {'n':>5s}")
    previous = ""
    for scene, node, option, mean, n in summary_rows(stats):
        if previous and scene != previous:
            print()
        previous = scene
        print(f"{scene:10s} {node:13s} {option:44s} {mean:8.3f} {n:5d}")


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)
    # Heavy imports live here so `--help` and linting never load model stacks.
    from discern.config import load_settings
    from discern.eval.datasets import DATA_DIR, load_dataset
    from discern.eval.metrics import match_image
    from discern.eval.runner import dataset_targets, load_rgb, tune_threshold
    from discern.experience.aggregate import build_memory, write_memory
    from discern.experience.harvest import (
        Configuration,
        detector_floors,
        fused_f1,
        harvest_image,
    )
    from discern.experience.harvest_cache import (
        CACHE_ROOT,
        HARVEST_POOL,
        load_cached_outputs,
        mapped_restorer,
    )
    from discern.experience.promotion import write_pointer
    from discern.experience.schema import ExperienceRecord, ExperienceStore
    from discern.models.registry import load_registry
    from discern.vision.stats import profile_from_stats

    settings = load_settings()
    exp = settings.thresholds.experience
    registry = load_registry()
    memory_dir = args.memory_dir or DATA_DIR / "memory"
    cache_root = args.cache_dir or CACHE_ROOT
    pool = args.detectors or list(HARVEST_POOL)
    revisions = {name: registry[name].revision for name in pool}
    grouping = settings.thresholds.grouping
    limit = args.limit or exp.harvest_samples_per_dataset
    adjudicate = args.adjudicate and not args.no_confirm
    if not adjudicate:
        print("WARNING: confirmation skipped, cheap fused scores stand in for adjudication")

    vlm: Any = None
    if adjudicate:
        from discern.models.loading import load_adapter

        vlm = load_adapter(registry[args.vlm])

    def adjudicated_f1(
        c: Configuration,
        image: Any,
        gt: Any,
        outs: Any,
        profile: Any,
        targets: list[str],
        floors: float | dict[str, float],
    ) -> float:
        from discern.agent.med import detect_image
        from discern.agent.schemas import DetectorInfo
        from discern.trace import TraceCollector

        cached = {n: _Cached(outs[c.variant][n]) for n in c.detectors}
        catalog = [
            DetectorInfo(
                name=n,
                capabilities=registry[n].role.replace("_", " "),
                speed_class=registry[n].speed_class,
            )
            for n in c.detectors
        ]
        final = detect_image(
            vlm,
            TraceCollector(),
            image,
            targets,
            profile,
            cached,  # type: ignore[arg-type]
            catalog,
            adjudicate_all=False,
            settings=settings,
            operating_thresholds=detector_floors(floors, c.detectors),  # as the cheap path
            priority=c.detectors,
        )
        return match_image(final, gt).f1

    records: list[ExperienceRecord] = []
    for dataset in args.datasets:
        images = load_dataset(dataset, "harvest")[:limit]
        targets = dataset_targets(images)
        mapped = {a.image_id: mapped_restorer(a.scene_label) for a in images}
        outputs = load_cached_outputs(cache_root, dataset, mapped, revisions)
        if args.min_score > 0:
            floors: float | dict[str, float] = args.min_score
        else:  # per-detector thresholds tuned on the harvest images (no restoration, no SR)
            floors = {
                name: tune_threshold(
                    {a.image_id: outputs[a.image_id][("none", "off")][name] for a in images}, images
                )
                for name in pool
            }
            print(f"[{dataset}] tuned detector thresholds: {floors}", flush=True)
        before = len(records)
        for a in images:
            image = load_rgb(a.path)
            profile = profile_from_stats(image)
            if a.scene_label is not None:  # harvest uses labelled data: the label is the truth
                profile = profile.model_copy(update={"scene_label": a.scene_label})
            outs = outputs[a.image_id]

            def confirm(
                c: Configuration,
                image: Any = image,
                gt: Any = a.objects,
                outs: Any = outs,
                profile: Any = profile,
                targets: list[str] = targets,
                floors: float | dict[str, float] = floors,
            ) -> float:
                if not adjudicate:
                    return fused_f1(c, image, gt, outs, grouping, floors)
                return adjudicated_f1(c, image, gt, outs, profile, targets, floors)

            records += harvest_image(
                a.image_id,
                profile,
                "detect",
                image,
                a.objects,
                outs,
                mapped[a.image_id],
                pool,
                grouping,
                confirm,
                exp.confirm_top_configs if adjudicate else 0,  # 0: no "adjudicated" rows
                args.version or "pending",
                floors,
            )
        print(f"[{dataset}] {len(images)} images, {len(records) - before} records", flush=True)

    version = args.version or f"{datetime.now(UTC):%Y%m%d}-{len(records)}"
    records = [r.model_copy(update={"memory_version": version}) for r in records]
    ExperienceStore(memory_dir / f"records-{version}.jsonl").append(records)
    memory = build_memory(records, version)
    print("wrote", write_memory(memory, memory_dir), f"({memory.version.record_count} records)")
    print_summary(memory.stats)
    if args.pin:
        pointer = memory_dir / settings.thresholds.serve.memory_pointer
        write_pointer(pointer, version)
        print("pinned", version, "in", pointer)
    else:
        print("not pinned (pass --pin to write the pointer)")


if __name__ == "__main__":
    main()
