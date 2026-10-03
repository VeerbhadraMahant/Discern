"""Build a DarkFace (low-light faces) subset from the HF mirror without downloading the 4.4GB zip.

Mirror: hieupth/dark_face (unofficial upload; license not stated; original is research-only).
"""

import random
from pathlib import Path

from PIL import Image
from remotezip import RemoteZip

from discern.eval.datasets import DATA_DIR, parse_darkface_label, save_annotations, split_subset
from discern.eval.types import AnnotatedImage

URL = "https://huggingface.co/datasets/hieupth/dark_face/resolve/main/dark_face_part001.zip"
N_GATE, N_HARVEST, SEED = 100, 50, 0
NAME = "darkface"


def main() -> None:
    ids = random.Random(SEED).sample(range(1, 6001), 240)
    items: list[AnnotatedImage] = []
    with RemoteZip(URL) as z:
        names = set(z.namelist())
        for i in ids:
            label, image = f"dark_face/label/{i}.txt", f"dark_face/image/{i}.png"
            if label not in names or image not in names:
                continue
            objects = parse_darkface_label(z.read(label).decode())
            if not objects:
                continue
            items.append(
                AnnotatedImage(
                    image_id=str(i),
                    path=Path("images") / f"{i}.png",
                    width=0,
                    height=0,
                    objects=objects,
                    scene_label="low_light",
                )
            )
        chosen = split_subset(items, N_GATE, N_HARVEST, SEED)
        out = []
        for a in chosen:
            dst = DATA_DIR / NAME / a.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(z.read(f"dark_face/image/{a.image_id}.png"))
            with Image.open(dst) as im:
                out.append(a.model_copy(update={"width": im.width, "height": im.height}))
    save_annotations(NAME, out)
    print(f"{NAME}: {len(items)} candidates, {len(out)} selected")


if __name__ == "__main__":
    main()
