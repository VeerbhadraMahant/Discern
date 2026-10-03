"""Restore one image with each restorer adapter and write side-by-sides to data/_demo/.

With --bench, also print peak VRAM and time per 1280x720 image for every adapter.
Usage: uv run python scripts/restore_demo.py [--bench] [--device cuda]
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

from discern.models.loading import load_adapter
from discern.models.registry import load_registry
from discern.models.roles import Image

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "_demo"
FOG_DIR = ROOT / "data" / "hazydet_real" / "images"
NIGHT_DIR = ROOT / "data" / "bdd100k_night" / "images"

# (registry entry, which sample it is applied to)
RESTORERS = [
    ("classical-dehaze", "fog"),
    ("classical-lowlight", "night"),
    ("classical-denoise", "night"),
    ("swinir-denoise", "night"),
    ("zero-dce-pp", "night"),
    ("llflow-lowlight", "night"),
    ("mprnet-derain", "fog"),
]
SR = "real-esrgan-x4plus"


def load(directory: Path, size: tuple[int, int] = (1280, 720)) -> Image:
    path = sorted(directory.glob("*.jpg"))[0]
    return np.asarray(PILImage.open(path).convert("RGB").resize(size), dtype=np.uint8)


def side_by_side(*images: Image) -> PILImage.Image:
    h = images[0].shape[0]
    resized = [
        PILImage.fromarray(im).resize((round(im.shape[1] * h / im.shape[0]), h)) for im in images
    ]
    canvas = PILImage.new("RGB", (sum(r.width for r in resized), h))
    x = 0
    for r in resized:
        canvas.paste(r, (x, 0))
        x += r.width
    return canvas


def measure(name: str, device: str, run, image: Image) -> tuple[Image, float, float]:  # type: ignore[no-untyped-def]
    """Load fresh, warm up once, then time one run. Returns output, seconds, peak GB."""
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    adapter = load_adapter(REGISTRY[name], device=device)
    run(adapter, image)  # warm-up (cudnn autotune, lazy init)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = run(adapter, image)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0.0
    del adapter
    return out, seconds, peak


REGISTRY = load_registry()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    samples = {"fog": load(FOG_DIR), "night": load(NIGHT_DIR)}

    for name, kind in RESTORERS:
        src = samples[kind]
        try:
            out, seconds, peak = measure(name, args.device, lambda a, im: a.restore(im), src)
        except Exception as exc:  # report and continue so one broken adapter does not hide the rest
            print(f"{name}: FAILED {type(exc).__name__}: {exc}")
            continue
        side_by_side(src, out).save(DEMO_DIR / f"{name}.jpg", quality=92)
        print(
            f"{name}: {seconds:.2f}s peak {peak:.2f} GB, mean {src.mean():.1f} -> {out.mean():.1f}"
        )

    # Super resolution on a 640x360 crop (2x4 -> 1280x720 / 2560x1440) and a 1080p tiled run.
    small = samples["fog"][:360, :640].copy()
    for factor in (2, 4):
        out, seconds, peak = measure(
            SR, args.device, lambda a, im, f=factor: a.upscale(im, f), small
        )
        side_by_side(np.asarray(PILImage.fromarray(small).resize(out.shape[1::-1])), out).save(
            DEMO_DIR / f"{SR}-x{factor}.jpg", quality=92
        )
        print(f"{SR} x{factor} on 640x360: {seconds:.2f}s peak {peak:.2f} GB")
    if args.bench:
        full = samples["fog"]
        out, seconds, peak = measure(SR, args.device, lambda a, im: a.upscale(im, 2), full)
        print(
            f"{SR} x2 on 1280x720 (-> {out.shape[1]}x{out.shape[0]}): {seconds:.2f}s {peak:.2f} GB"
        )
        big = np.asarray(PILImage.fromarray(full).resize((1920, 1080)))
        out, seconds, peak = measure(SR, args.device, lambda a, im: a.upscale(im, 2), big)
        print(f"{SR} x2 on 1920x1080: {seconds:.2f}s peak {peak:.2f} GB")


if __name__ == "__main__":
    main()
