"""Build the HazyDet real-world (natural haze) subset from the HF mirror.

Mirror: ironmanfcf/HazyDet (cc-by-nc-4.0 on the card; unofficial upload of the GrokCV dataset).
Pools real_world train+test (600 frames) and draws disjoint gate/harvest subsets.
"""

import json
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from discern.eval.datasets import DATA_DIR, coco_to_annotated, save_annotations, split_subset

N_GATE, N_HARVEST, SEED = 100, 50, 0
NAME = "hazydet_real"


def main() -> None:
    zpath = hf_hub_download("ironmanfcf/HazyDet", "real_world.zip", repo_type="dataset")
    items = []
    with zipfile.ZipFile(zpath) as z:
        for part in ("train", "test"):
            coco = json.loads(z.read(f"real_world/{part}_real.json"))
            items += coco_to_annotated(
                coco,
                Path("images"),
                {"car": "car", "truck": "truck", "bus": "bus"},
                "fog",
                id_prefix=f"{part}_",
            )
        chosen = split_subset(items, N_GATE, N_HARVEST, SEED)
        for a in chosen:
            part = a.image_id.split("_", 1)[0]
            dst = DATA_DIR / NAME / a.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(f"real_world/{part}/{a.path.name}") as src, dst.open("wb") as out:
                shutil.copyfileobj(src, out)
    save_annotations(NAME, chosen)
    print(f"{NAME}: {len(items)} candidates, {len(chosen)} selected")


if __name__ == "__main__":
    main()
