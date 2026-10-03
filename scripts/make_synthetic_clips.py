"""Build synthetic video clips from the labelled BDD100K gate images.

Each clip is a slow zoom from the full still image to a centred crop (default 0.85 of the frame)
at 10 fps for 8 seconds, so every object inside the final crop is visible for the whole clip and
the true number of distinct objects per label is known (`discern.eval.synth_clips`).

Writes data/clips/<set>/<image_id>.mp4 and data/clips/manifest.json. Sets:
  bdd100k_clear, bdd100k_night, bdd100k_rainy   clips of the real images, no degradation
  bdd100k_clear_fog   (with --fog) the clear-day images with SYNTHETIC fog, severity 0.6, from
                      discern.eval.degrade (atmospheric scattering with a vertical depth prior).
                      This fog is simulated; it is not real fog footage.

Usage: uv run python scripts/make_synthetic_clips.py [--per-dataset 6] [--fog] [--out-dir DIR]
The manifest is merged by clip set: re-running a set replaces only that set's entries.
"""

import argparse
import json
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from discern.config import load_settings
from discern.eval.datasets import DATA_DIR, load_dataset
from discern.eval.synth_clips import (
    DEFAULT_END_SCALE,
    DEFAULT_MIN_VISIBLE_FRACTION,
    ClipEntry,
    ClipObject,
    Degradation,
    Manifest,
    clip_seed,
    expected_counts,
    final_crop,
    make_zoom_clip,
    select_images,
)
from discern.eval.types import AnnotatedImage
from discern.models.roles import Image

DATASETS = ("bdd100k_clear", "bdd100k_night", "bdd100k_rainy")
FOG = Degradation("fog", 0.6)
FOG_SET = "bdd100k_clear_fog"
SECONDS = 8
FPS = 10


def encode_clip(frames: Sequence[Image], fps: int, dst: Path) -> str:
    """Write frames as H.264 (yuv420p), or MPEG-4 part 2 when this PyAV build has no libx264.
    Returns the codec used. Width and height are cut down to even numbers."""
    height, width = frames[0].shape[0] & ~1, frames[0].shape[1] & ~1
    last_error: Exception | None = None
    for codec in ("libx264", "mpeg4"):
        try:
            with av.open(str(dst), "w") as out:
                stream = out.add_stream(codec, rate=Fraction(fps))
                stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
                for frame in frames:
                    video = av.VideoFrame.from_ndarray(
                        np.ascontiguousarray(frame[:height, :width]), format="rgb24"
                    )
                    for packet in stream.encode(video):
                        out.mux(packet)
                for packet in stream.encode():
                    out.mux(packet)
            return codec
        except (av.error.FFmpegError, ValueError) as err:
            last_error = err
            dst.unlink(missing_ok=True)
    raise RuntimeError(f"no usable video encoder: {last_error}")


def load_rgb(a: AnnotatedImage) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(a.path) as pil:
        return np.asarray(pil.convert("RGB"), dtype=np.uint8)


def build_set(
    name: str,
    source: str,
    degradation: Degradation | None,
    per_dataset: int,
    out_dir: Path,
    end_scale: float,
    min_fraction: float,
) -> list[ClipEntry]:
    cfg = load_settings().thresholds.degrade
    images = select_images(
        load_dataset(source, "gate"), per_dataset, end_scale=end_scale,
        min_visible_fraction=min_fraction,
    )
    if len(images) < per_dataset:
        print(f"warning: {source} has only {len(images)} eligible images", flush=True)
    entries: list[ClipEntry] = []
    for a in images:
        rgb = load_rgb(a)
        height, width = rgb.shape[:2]
        n_frames = SECONDS * FPS
        seed = clip_seed(a.image_id)
        frames, _ = make_zoom_clip(
            rgb, a.objects, n_frames, float(FPS), end_scale,
            min_visible_fraction=min_fraction, degradation=degradation, cfg=cfg, seed=seed,
        )
        rel = f"{name}/{a.image_id}.mp4"
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        codec = encode_clip(frames, FPS, dst)
        crop = final_crop(width, height, end_scale)
        entries.append(
            ClipEntry(
                clip=rel,
                name=name,
                source_dataset=source,
                image_id=a.image_id,
                degradation=degradation,
                expected_counts=expected_counts(a.objects, crop, min_fraction),
                final_crop=crop,
                fps=float(FPS),
                duration_seconds=n_frames / FPS,
                n_frames=n_frames,
                width=width,
                height=height,
                end_scale=end_scale,
                min_visible_fraction=min_fraction,
                seed=seed,
                gt_objects=[ClipObject(label=o.label, box=o.box) for o in a.objects],
            )
        )
        print(f"{rel} [{codec}] expected={entries[-1].expected_counts}", flush=True)
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--per-dataset", type=int, default=6)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--fog", action="store_true", help=f"also build {FOG_SET} (synthetic fog 0.6)")
    ap.add_argument("--end-scale", type=float, default=DEFAULT_END_SCALE)
    ap.add_argument("--min-visible-fraction", type=float, default=DEFAULT_MIN_VISIBLE_FRACTION)
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR / "clips")
    args = ap.parse_args()

    jobs: list[tuple[str, str, Degradation | None]] = [(d, d, None) for d in args.datasets]
    if args.fog:
        jobs.append((FOG_SET, "bdd100k_clear", FOG))
    manifest_path = args.out_dir / "manifest.json"
    kept: list[ClipEntry] = []
    if manifest_path.exists():
        kept = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8")).clips
    rebuilt = {name for name, _, _ in jobs}
    clips = [c for c in kept if c.name not in rebuilt]
    for name, source, degradation in jobs:
        clips += build_set(
            name, source, degradation, args.per_dataset, args.out_dir, args.end_scale,
            args.min_visible_fraction,
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(Manifest(clips=clips).model_dump(mode="json"), indent=1), encoding="utf-8"
    )
    print(f"wrote {len(clips)} clips to {manifest_path}")


if __name__ == "__main__":
    main()
