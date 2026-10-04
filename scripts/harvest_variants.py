"""Milestone 9 harvest, step 1: run the real models and cache detections per variant.

For each degraded dataset and each image of its HARVEST split this builds the variants the harvest
enumerates, restorer in {none, mapped} x super-resolution in {off, auto}, and runs every detector
of the pool on each variant. The mapped restorer comes from the dataset-implied scene label
(fog -> dehaze, low_light -> lowlight, rain -> derain). SR "auto" is Real-ESRGAN at the integer
factor that reaches `agent.sr_target_long_side`; images that already reach it have no auto
variant (auto equals off). Boxes are mapped back to the original frame. The cache format and its
reader are documented in `discern.experience.harvest_cache`.

Usage: uv run python scripts/harvest_variants.py [--limit N] [dataset ...]
Then:  uv run python scripts/harvest.py [--pin] [dataset ...]

Reruns resume: caches are keyed by (dataset, variant, detector, detector revision) under
data/_cache/seeh/. Phases per dataset keep VRAM low: restorers and super-resolution first (each
model evicted when done), then one detector at a time. No VLM is needed. GPU only; never run by
the tests.
"""

import argparse
import gc
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from discern.agent.nodes.restorer_select import NONE
from discern.config import load_settings
from discern.config.settings import Settings
from discern.eval.datasets import load_dataset
from discern.eval.runner import dataset_targets, load_rgb
from discern.eval.types import AnnotatedImage
from discern.experience.harvest import SR_AUTO, SR_OFF, VariantKey
from discern.experience.harvest_cache import (
    CACHE_ROOT,
    HARVEST_POOL,
    HARVEST_RESTORERS,
    SR_ENTRY,
    CacheKey,
    fill_cache,
    mapped_restorer,
    pending,
    sr_factor,
    variant_keys,
)
from discern.models.loading import load_adapter
from discern.models.manager import ModelManager, RegistryEntry
from discern.models.registry import load_registry
from discern.models.roles import Detection

DEGRADED = ["hazydet_real", "bdd100k_night", "bdd100k_rainy", "darkface"]
Todo = dict[VariantKey, list[tuple[str, int | None]]]


def free_gpu(model: object) -> None:
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def image_size(path: Path) -> tuple[int, int]:
    with PILImage.open(path) as im:
        return im.size


def variant_path(tmp: Path, a: AnnotatedImage, variant: VariantKey) -> Path:
    """Where a variant's image lives: the original for (none, off), else a temporary PNG."""
    if variant == (NONE, SR_OFF):
        return a.path
    return tmp / f"{variant[0]}-{variant[1]}" / f"{a.image_id}.png"


def build_todo(
    images: Sequence[AnnotatedImage], mapped: Mapping[str, str], target: int
) -> Todo:
    """Per variant, the (image id, SR factor | None) pairs to run. The restorer variants only
    cover images whose mapped restorer it is; factor 1 means no SR, None means auto is skipped."""
    sizes = {a.image_id: image_size(a.path) for a in images}
    todo: Todo = {}
    for variant in sorted({v for m in mapped.values() for v in variant_keys(m)}):
        restorer, sr = variant
        todo[variant] = [
            (a.image_id, 1 if sr == SR_OFF else sr_factor(*sizes[a.image_id], target))
            for a in images
            if restorer == NONE or mapped[a.image_id] == restorer
        ]
    return todo


def materialize(
    manager: ModelManager,
    images: Sequence[AnnotatedImage],
    todo: Todo,
    needed: Sequence[VariantKey],
    tmp: Path,
) -> None:
    """Phase 1: write the restored and super-resolved images that some detector still needs."""
    by_id = {a.image_id: a for a in images}
    factors = {v: dict(entries) for v, entries in todo.items()}
    for restorer in sorted({r for r, _ in needed}):
        for image_id in sorted({i for v in needed if v[0] == restorer for i, _ in todo[v]}):
            a = by_id[image_id]
            jobs: dict[str, int] = {}
            for variant in needed:
                factor = factors[variant].get(image_id)
                missing = not variant_path(tmp, a, variant).exists()
                if variant[0] == restorer and factor is not None and missing:
                    jobs[variant[1]] = factor
            if restorer == NONE:
                jobs.pop(SR_OFF, None)  # the original is used as is
            if not jobs:
                continue
            rgb = load_rgb(a.path)
            if restorer == NONE:
                out = rgb
            else:
                model: Any = manager.get(f"restorer_{restorer}", HARVEST_RESTORERS[restorer])
                out = model.restore(rgb)
            for sr, factor in jobs.items():
                image = out
                if sr == SR_AUTO:
                    sr_model: Any = manager.get("super_resolver", SR_ENTRY)
                    image = sr_model.upscale(out, factor)
                dest = variant_path(tmp, a, (restorer, sr))
                dest.parent.mkdir(parents=True, exist_ok=True)
                PILImage.fromarray(np.asarray(image)).save(dest, compress_level=1)
        if restorer != NONE:
            manager.evict(HARVEST_RESTORERS[restorer])
    manager.evict(SR_ENTRY)


def run_detector(
    detector: Any,
    key: CacheKey,
    entries: Sequence[tuple[str, int | None]],
    paths: Mapping[str, Path],
    targets: Sequence[str],
) -> None:
    def detect(image_id: str) -> list[Detection]:
        found: list[Detection] = detector.detect(load_rgb(paths[image_id]), targets)
        return found

    fill_cache(key, entries, detect)


def run_dataset(
    name: str,
    manager: ModelManager,
    registry: Mapping[str, RegistryEntry],
    settings: Settings,
    limit: int,
    detectors: Sequence[str],
    root: Path,
    keep_images: bool,
) -> None:
    t0 = time.perf_counter()
    images = load_dataset(name, "harvest")[:limit]
    targets = dataset_targets(images)
    mapped = {a.image_id: mapped_restorer(a.scene_label) for a in images}
    unknown = {m for m in mapped.values() if m != NONE and m not in HARVEST_RESTORERS}
    if unknown:
        raise SystemExit(f"{name}: no harvest restorer entry for {sorted(unknown)}")
    todo = build_todo(images, mapped, settings.thresholds.agent.sr_target_long_side)
    by_id = {a.image_id: a for a in images}
    tmp = root / name / "_images"
    print(
        f"[{name}] {len(images)} images, restorers {sorted(set(mapped.values()))}, "
        f"variants {sorted(todo)}",
        flush=True,
    )

    def key(variant: VariantKey, detector: str) -> CacheKey:
        return CacheKey(root, name, variant, detector, registry[detector].revision)

    def pending_for(variant: VariantKey, detector: str) -> bool:
        ids = [i for i, _ in todo[variant]]
        return bool(pending(key(variant, detector), ids))

    needed = [v for v in todo if any(pending_for(v, d) for d in detectors)]
    if not needed:
        print(f"[{name}] all caches complete", flush=True)
        return

    t1 = time.perf_counter()
    materialize(manager, images, todo, needed, tmp)
    print(f"[{name}] restore/SR phase {time.perf_counter() - t1:.0f}s", flush=True)

    for detector_name in detectors:
        variants = [v for v in todo if pending_for(v, detector_name)]
        if not variants:
            continue
        t2 = time.perf_counter()
        detector: Any = manager.get(registry[detector_name].role, detector_name)
        for variant in variants:
            paths = {i: variant_path(tmp, by_id[i], variant) for i, _ in todo[variant]}
            run_detector(detector, key(variant, detector_name), todo[variant], paths, targets)
            print(f"[{name}] {detector_name} {variant} done", flush=True)
        manager.evict(detector_name)
        print(f"[{name}] {detector_name} {time.perf_counter() - t2:.0f}s", flush=True)

    if not keep_images:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{name}] finished in {time.perf_counter() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("datasets", nargs="*", default=DEGRADED)
    ap.add_argument("--detectors", nargs="+", default=list(HARVEST_POOL))
    ap.add_argument("--limit", type=int, default=None, help="images per dataset (default: 50)")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)
    ap.add_argument("--keep-images", action="store_true", help="keep the temporary variant images")
    args = ap.parse_args()

    settings = load_settings()
    registry = load_registry()
    limit = args.limit or settings.thresholds.experience.harvest_samples_per_dataset
    manager = ModelManager(registry, {}, settings.profile.vram_budget_gb, load_adapter, free_gpu)
    for dataset in args.datasets:
        run_dataset(
            dataset,
            manager,
            registry,
            settings,
            limit,
            args.detectors,
            args.cache_dir,
            args.keep_images,
        )


if __name__ == "__main__":
    main()
