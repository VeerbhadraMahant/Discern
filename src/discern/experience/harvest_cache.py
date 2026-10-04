"""Cached detector outputs for the offline harvest (system-design 5.7 step 2).

`scripts/harvest_variants.py` writes them with real models; `scripts/harvest.py` reads them back
into the `CachedOutputs` shape that `discern.experience.harvest` scores. No model runs here.

Cache format (one JSON file per dataset, variant, detector and detector revision, at
`<root>/<dataset>/<restorer>-<sr>-<detector>-<revision[:10]>.json`):

    {"format": 1, "dataset": ..., "restorer": ..., "sr": ..., "detector": ..., "revision": ...,
     "detections": {image_id: [{"box": [x1, y1, x2, y2], "label": ..., "score": ...,
                                "detector": ...}, ...]},
     "skipped": [image_id, ...]}

Boxes are absolute pixels in the ORIGINAL frame (boxes from super-resolved images are already
scaled back). `skipped` lists images for which the "auto" super-resolution variant does not exist
because the image already reaches the target size: for them auto equals off, and the loader reuses
the off outputs. A variant a mapped restorer does not apply to has no file and is never read.
"""

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import TypeAdapter

from discern.agent.nodes.restorer_select import NONE, RESTORER_FOR_SCENE
from discern.agent.nodes.sr_select import required_factor
from discern.eval.datasets import DATA_DIR
from discern.experience.harvest import SR_AUTO, SR_OFF, CachedOutputs, VariantKey
from discern.models.roles import Detection
from discern.vision.boxes import scale

CACHE_FORMAT = 1
CACHE_ROOT = DATA_DIR / "_cache" / "seeh"
HARVEST_POOL = ("yolo-world-v2", "owlv2-base", "grounding-dino-base")
# Restorer role name (as `RESTORER_FOR_SCENE` gives it) -> registry entry used for the harvest.
HARVEST_RESTORERS = {
    "dehaze": "classical-dehaze",
    "lowlight": "zero-dce-pp",
    "derain": "mprnet-derain",
    "denoise": "swinir-denoise",
}
SR_ENTRY = "real-esrgan-x4plus"

_SCENE_RESTORER: Mapping[str, str] = {str(k): v for k, v in RESTORER_FOR_SCENE.items()}
_DETECTIONS = TypeAdapter(list[Detection])


def mapped_restorer(scene_label: str | None) -> str:
    """The restorer the rule map gives for a (dataset-implied) scene label; none if unmapped."""
    return _SCENE_RESTORER.get(scene_label or "", NONE)


def variant_keys(mapped: str) -> list[VariantKey]:
    """Restorer in {none, mapped} x SR in {off, auto}."""
    restorers = [NONE] if mapped == NONE else [NONE, mapped]
    return [(r, s) for r in restorers for s in (SR_OFF, SR_AUTO)]


def sr_factor(width: int, height: int, target_long_side: int) -> int | None:
    """Integer SR factor of the "auto" option, or None when the image already reaches the target
    (then auto equals off and the variant is skipped)."""
    return required_factor(max(width, height), target_long_side)


def scale_back(detections: Sequence[Detection], factor: int) -> list[Detection]:
    """Map boxes from the factor-times-larger processed image back to the original frame."""
    if factor == 1:
        return list(detections)
    return [d.model_copy(update={"box": scale(d.box, 1.0 / factor)}) for d in detections]


@dataclass(frozen=True)
class CacheKey:
    root: Path
    dataset: str
    variant: VariantKey
    detector: str
    revision: str

    @property
    def path(self) -> Path:
        restorer, sr = self.variant
        name = f"{restorer}-{sr}-{self.detector}-{self.revision[:10]}.json"
        return self.root / self.dataset / name


@dataclass
class VariantCache:
    detections: dict[str, list[Detection]] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)

    def has(self, image_id: str) -> bool:
        return image_id in self.detections or image_id in self.skipped


def write_cache(key: CacheKey, cache: VariantCache) -> None:
    restorer, sr = key.variant
    payload = {
        "format": CACHE_FORMAT,
        "dataset": key.dataset,
        "restorer": restorer,
        "sr": sr,
        "detector": key.detector,
        "revision": key.revision,
        "detections": {
            k: _DETECTIONS.dump_python(v, mode="json") for k, v in cache.detections.items()
        },
        "skipped": sorted(cache.skipped),
    }
    key.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = key.path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, key.path)  # an interrupted write never leaves a truncated cache


def read_cache(key: CacheKey) -> VariantCache | None:
    if not key.path.exists():
        return None
    raw = json.loads(key.path.read_text(encoding="utf-8"))
    if raw.get("format") != CACHE_FORMAT:
        raise ValueError(f"{key.path}: cache format {raw.get('format')}, expected {CACHE_FORMAT}")
    return VariantCache(
        detections={k: _DETECTIONS.validate_python(v) for k, v in raw["detections"].items()},
        skipped=set(raw["skipped"]),
    )


def pending(key: CacheKey, image_ids: Sequence[str]) -> list[str]:
    """Image ids not yet in the cache (all of them when there is no cache file)."""
    cache = read_cache(key) or VariantCache()
    return [i for i in image_ids if not cache.has(i)]


def fill_cache(
    key: CacheKey,
    todo: Sequence[tuple[str, int | None]],
    detect: Callable[[str], list[Detection]],
    save_every: int = 10,
) -> VariantCache:
    """Detect every image of `todo` not yet cached and persist the result (resumable).

    `todo` pairs an image id with the SR factor of its processed image (1 when SR is off) or None
    when the variant does not exist for it (SR auto on an image that already reaches the target).
    `detect` returns detections in the processed image's pixels; they are stored in the original
    frame.
    """
    cache = read_cache(key) or VariantCache()
    done = 0
    for image_id, factor in todo:
        if cache.has(image_id):
            continue
        if factor is None:
            cache.skipped.add(image_id)
        else:
            cache.detections[image_id] = scale_back(detect(image_id), factor)
        done += 1
        if done % save_every == 0:
            write_cache(key, cache)
    write_cache(key, cache)
    return cache


def load_cached_outputs(
    root: Path,
    dataset: str,
    mapped: Mapping[str, str],
    detectors: Mapping[str, str],
) -> dict[str, CachedOutputs]:
    """Per image, `variant -> detector -> detections` as `harvest_image` expects.

    `mapped` gives each image id's mapped restorer, `detectors` maps detector name to registry
    revision. Raises when a cache is missing or lacks an image (run `scripts/harvest_variants.py`).
    """
    caches: dict[tuple[VariantKey, str], VariantCache] = {}
    for variant in {v for m in mapped.values() for v in variant_keys(m)}:
        for name, revision in detectors.items():
            key = CacheKey(root, dataset, variant, name, revision)
            cache = read_cache(key)
            if cache is None:
                raise FileNotFoundError(f"{key.path} is missing; run scripts/harvest_variants.py")
            caches[(variant, name)] = cache

    outputs: dict[str, CachedOutputs] = {}
    for image_id, restorer in mapped.items():
        per_variant: dict[VariantKey, dict[str, list[Detection]]] = {}
        for variant in variant_keys(restorer):
            per_variant[variant] = {}
            for name in detectors:
                cache = caches[(variant, name)]
                source = cache
                if image_id in cache.skipped:  # auto equals off for this image
                    source = caches[((variant[0], SR_OFF), name)]
                if image_id not in source.detections:
                    raise ValueError(
                        f"{dataset}: no cached {variant}/{name} output for {image_id}; "
                        "run scripts/harvest_variants.py"
                    )
                per_variant[variant][name] = source.detections[image_id]
        outputs[image_id] = per_variant
    return outputs
