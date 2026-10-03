"""Write configs/gate_subset.json: the fixed gate and harvest image IDs per dataset.

The data itself is gitignored; this manifest is the versioned record of which images
count as the gate subset (reported numbers) and the harvest pool (tuning, SEEH).
"""

import json
from pathlib import Path

from discern.eval.datasets import load_dataset

DATASETS = [
    "coco_val_clean",
    "bdd100k_clear",
    "bdd100k_rainy",
    "bdd100k_night",
    "hazydet_real",
    "darkface",
]
OUT = Path(__file__).resolve().parents[1] / "configs" / "gate_subset.json"

manifest = {
    name: {
        "gate": sorted(a.image_id for a in load_dataset(name, "gate")),
        "harvest": sorted(a.image_id for a in load_dataset(name, "harvest")),
    }
    for name in DATASETS
}
OUT.write_text(json.dumps(manifest, indent=1) + "\n")
print(f"wrote {OUT}")
