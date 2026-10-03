"""Build the BDD100K night, rainy and clear-day subsets from the HF FiftyOne mirror.

Mirror: dgural/bdd100k (unofficial upload of the BDD100K validation split, 10k images).
Original data license: BDD100K is for non-commercial research use.
Writes data/bdd100k_{night,rainy,clear}/{images,annotations.json}. Nothing here is committed.
"""

import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from discern.eval.datasets import DATA_DIR, bdd_sample_to_annotated, save_annotations, split_subset

REPO = "dgural/bdd100k"
N_GATE, N_HARVEST, SEED = 100, 50, 0

SUBSETS = {
    # name: (scene_label, predicate on (weather, timeofday))
    "bdd100k_night": ("low_light", lambda w, t: t == "night" and w in {"clear", "overcast"}),
    "bdd100k_rainy": ("rain", lambda w, t: w == "rainy" and t == "daytime"),
    "bdd100k_clear": ("normal", lambda w, t: w == "clear" and t == "daytime"),
}


def main() -> None:
    samples_json = Path(DATA_DIR / "bdd100k" / "samples.json")
    samples = json.loads(samples_json.read_text())["samples"]
    for name, (scene_label, keep) in SUBSETS.items():
        candidates = []
        for s in samples:
            if not keep(s["weather"]["label"], s["timeofday"]["label"]):
                continue
            a = bdd_sample_to_annotated(s, scene_label, Path("images") / Path(s["filepath"]).name)
            if a is not None:
                candidates.append((s["filepath"], a))
        by_id = {a.image_id: fp for fp, a in candidates}
        chosen = split_subset([a for _, a in candidates], N_GATE, N_HARVEST, SEED)
        for a in chosen:
            src = hf_hub_download(REPO, by_id[a.image_id], repo_type="dataset")
            dst = DATA_DIR / name / a.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        save_annotations(name, chosen)
        print(f"{name}: {len(candidates)} candidates, {len(chosen)} selected")


if __name__ == "__main__":
    main()
