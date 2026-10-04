"""Run the real serving engine on a few gate images: clean, detect, annotate (a manual smoke test).

Usage: uv run python scripts/smoke_engine.py [dataset] [n_images] [targets ...]
Writes annotated images to data/_demo/engine/ and prints the plan, detections and timings.
"""

import sys
import time
from pathlib import Path

from PIL import Image as PILImage

from discern.eval.datasets import DATA_DIR, load_dataset
from discern.eval.runner import dataset_targets
from discern.serve.engine import build_engine


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "bdd100k_night"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    images = load_dataset(dataset, "gate")[:n]
    targets = sys.argv[3:] or dataset_targets(images)
    out_dir = DATA_DIR / "_demo" / "engine"
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = build_engine()
    for a in images:
        t0 = time.perf_counter()
        cleaned = engine.clean_image(a.path)
        t1 = time.perf_counter()
        result = engine.detect_targets(cleaned, targets)
        t2 = time.perf_counter()
        dest = Path(out_dir / f"{dataset}-{a.image_id}.png")
        PILImage.fromarray(result.annotated).save(dest)
        print(
            f"{a.image_id}: scene={cleaned.profile.scene_label} plan={cleaned.plan.decisions} "
            f"detections={len(result.detections)} clean={t1 - t0:.1f}s "
            f"detect={t2 - t1:.1f}s -> {dest}"
        )
    print("measured GPU seconds:", engine.measured_gpu_seconds())


if __name__ == "__main__":
    main()
