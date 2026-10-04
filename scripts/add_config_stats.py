"""Rewrite a memory file with configuration stats rebuilt from its raw records (no re-harvest).

Reads data/memory/records-<version>.jsonl and rewrites data/memory/memory-<version>.json in place
with the same version id, adding the per-configuration stats the joint experience policy needs.
Node-level stats, the version and the source hash are unchanged.

Usage: uv run python scripts/add_config_stats.py [--version ID] [--memory-dir DIR]
(default version: the one in the pinned pointer)
"""

import argparse
import sys
from pathlib import Path

from discern.config import load_settings
from discern.eval.datasets import DATA_DIR
from discern.experience.aggregate import (
    load_memory,
    memory_path,
    with_configuration_stats,
    write_memory,
)
from discern.experience.promotion import read_pointer
from discern.experience.schema import ExperienceStore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=None, help="default: the pinned version")
    ap.add_argument("--memory-dir", type=Path, default=DATA_DIR / "memory")
    args = ap.parse_args()
    pointer = args.memory_dir / load_settings().thresholds.serve.memory_pointer
    version = args.version or read_pointer(pointer)
    if not version:
        sys.exit("no memory version: pin one (harvest.py --pin) or pass --version")
    records = ExperienceStore(args.memory_dir / f"records-{version}.jsonl").load()
    if not records:
        sys.exit(f"no records found for version {version} in {args.memory_dir}")
    memory = with_configuration_stats(load_memory(memory_path(args.memory_dir, version)), records)
    path = write_memory(memory, args.memory_dir)
    print("wrote", path, f"({len(memory.config_stats)} config stats)")


if __name__ == "__main__":
    main()
