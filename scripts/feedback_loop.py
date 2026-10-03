"""Scheduled local job: opt-in feedback -> labelled samples -> harvest -> candidate memory ->
gate -> promote (system-design 5.7 online growth).

Usage:
  uv run python scripts/feedback_loop.py --feedback-dir <private dir> --version v2 \
      --metrics candidate.json --baseline baseline.json [--memory-dir data/memory]

Status: skeleton. Steps 1, 3 and 4 run today. Step 2 (harvest of the new samples) reuses the
per-image routines in `discern.experience.harvest` and the wiring in `scripts/harvest.py`; it
needs the GPU models and is not connected yet, so it raises until it is. Candidate and baseline
metrics are JSON files of gate metric name -> value, produced by gate runs for the candidate and
for the pinned memory version.
"""

import argparse
import json
from pathlib import Path


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--feedback-dir", type=Path, required=True, help="private feedback store")
    p.add_argument("--version", required=True, help="candidate memory version id")
    p.add_argument("--metrics", type=Path, required=True, help="gate metrics of the candidate")
    p.add_argument("--baseline", type=Path, required=True, help="gate metrics of the pinned memory")
    p.add_argument("--memory-dir", type=Path, default=Path("data/memory"))
    return p.parse_args(argv)


def _metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items()}


def _harvest(sample_count: int, memory_dir: Path, version: str) -> None:
    """Harvest feedback samples into records (source="feedback") appended to
    `memory_dir/records.jsonl`; wire like scripts/harvest.py once the GPU pass is available."""
    raise NotImplementedError(f"harvest of {sample_count} feedback samples is not wired yet")


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    # Heavy imports live here so `--help` and linting never load model stacks.
    from discern.config import load_settings
    from discern.experience.aggregate import build_memory, write_memory
    from discern.experience.promotion import promote_candidate
    from discern.experience.schema import ExperienceStore
    from discern.feedback.schema import FeedbackStore, feedback_to_labelled_samples

    settings = load_settings()
    store = FeedbackStore(args.feedback_dir)

    # 1. feedback -> labelled samples (opt-in only)
    samples = feedback_to_labelled_samples(store.load(), store)
    print(f"{len(samples)} labelled samples from opt-in feedback")
    if not samples:
        return 0

    # 2. harvest
    _harvest(len(samples), args.memory_dir, args.version)

    # 3. candidate memory version from all records
    records = ExperienceStore(args.memory_dir / "records.jsonl").load()
    memory = build_memory(records, args.version)
    print("candidate:", write_memory(memory, args.memory_dir))

    # 4. gate and promote
    decision = promote_candidate(
        args.version,
        _metrics(args.metrics),
        _metrics(args.baseline),
        settings.thresholds.eval_gate,
        args.memory_dir / "pinned.json",
    )
    for reason in decision.reasons:
        print("  -", reason)
    print("PROMOTED" if decision.promote else "NOT PROMOTED")
    return 0 if decision.promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
