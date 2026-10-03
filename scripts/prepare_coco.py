"""Build a clean-reference subset from COCO val2017 (annotations CC BY 4.0)."""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from remotezip import RemoteZip

from discern.eval.datasets import DATA_DIR, coco_to_annotated, save_annotations, split_subset

N_GATE, N_HARVEST, SEED = 100, 50, 0
NAME = "coco_val_clean"
ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
LABELS = {
    "person": "person",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
}


def fetch(url: str, dst: Path, attempts: int = 4) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                tmp = dst.with_suffix(".part")
                tmp.write_bytes(r.read())
                tmp.replace(dst)
            return
        except OSError:
            if attempt == attempts - 1:
                raise


def main() -> None:
    with RemoteZip(ANN_ZIP) as z:
        coco = json.loads(z.read("annotations/instances_val2017.json"))
    items = coco_to_annotated(coco, Path("images"), LABELS, "normal")
    chosen = split_subset(items, N_GATE, N_HARVEST, SEED)
    with ThreadPoolExecutor(8) as pool:
        for a in chosen:
            dst = DATA_DIR / NAME / a.path
            if not dst.exists():
                pool.submit(fetch, f"http://images.cocodataset.org/val2017/{a.path.name}", dst)
    save_annotations(NAME, chosen)
    print(f"{NAME}: {len(items)} candidates, {len(chosen)} selected")


if __name__ == "__main__":
    main()
