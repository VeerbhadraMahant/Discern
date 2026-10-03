"""Assemble the Hugging Face Space bundle: src/discern, configs/, the classical restorers from
legacy/v0 and the files in space/, in the repository layout so config paths resolve unchanged.
Data, weights and caches are never copied. Nothing is uploaded here; see deploy-space.yml.

Usage: uv run python scripts/build_space_bundle.py [--out build/space] [--clean]
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = ("__pycache__", ".mypy_cache", ".pytest_cache", "data", "weights", "mlruns")
EXCLUDED_SUFFIXES = (".pyc", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin")
# (source relative to the repository, destination relative to the bundle)
COPIES = (
    ("src/discern", "src/discern"),
    ("configs", "configs"),
    ("legacy/v0/restoration.py", "legacy/v0/restoration.py"),
)


def _ignore(directory: str, names: list[str]) -> set[str]:
    return {n for n in names if n in EXCLUDED_DIRS or n.endswith(EXCLUDED_SUFFIXES)}


def build_bundle(root: Path, out: Path) -> list[Path]:
    """Copy the bundle into `out` (which must not exist); returns the files written."""
    if out.exists():
        raise FileExistsError(f"{out} exists; pass --clean to replace it")
    for source, destination in COPIES:
        src, dst = root / source, out / destination
        if src.is_dir():
            shutil.copytree(src, dst, ignore=_ignore)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for item in sorted((root / "space").iterdir()):
        if item.is_file():
            shutil.copy2(item, out / item.name)
    return sorted(p for p in out.rglob("*") if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "space")
    ap.add_argument("--clean", action="store_true", help="delete --out first if it exists")
    args = ap.parse_args()
    out = args.out.resolve()
    if args.clean and out.exists():
        if (ROOT / "build") in out.parents:
            shutil.rmtree(out)
        else:
            print(f"refusing to delete {out}: it is not inside {ROOT / 'build'}", file=sys.stderr)
            return 1
    files = build_bundle(ROOT, out)
    print(f"wrote {len(files)} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
