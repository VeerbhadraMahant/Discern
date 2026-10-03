"""Real-model smoke evaluation of the video pipeline on synthetic clips (needs the GPU).

Run `scripts/make_synthetic_clips.py` first. For every clip in data/clips/manifest.json this loads
the pipeline exactly as the serving engine does (`discern.serve.engine.build_engine`) and runs:
  ingest   shots, per-shot SAIR plan, sampling, and the video index (`Engine.ingest`)
  count    "How many <label> are there?" for every label with an expected count of at least 1;
           the first such question runs the track stage (fast MED detectors + grouping, ByteTrack,
           re-identification, per-track VLM adjudication), later ones reuse the scan per label
  locate   "Where is the first <label>?" once, for the most numerous label; its tracks are scored
           with box F1@0.5 against the per-frame ground truth of that label (class-agnostic inside
           the one label, because the question names it)
The answer text is parsed for an integer and compared with the expected count (exact match and
MAE). The ModelManager evicts least-recently-used models inside the profile VRAM budget.

gpu_seconds_per_video_second is, per clip, the wall time of the engine's GPU-decorated calls
(ingest, index, every question) over the video duration, taken from the trace; when no CUDA is
visible those calls are not recorded as GPU time and the clip's total wall time is used instead.
`wall_seconds_per_video_second` is always the total wall time. A clip that raises is logged and
skipped: it counts in `clips_failed` and in no other metric.

Metrics are logged to the MLflow experiment named by configs eval_gate.experiment (sqlite at
mlruns/mlflow.db): one run per clip set, then a final run over all clips (run name
video-smoke/all) so that scripts/run_gate.py, which reads the latest finished run, sees the
overall numbers. temporal_iou (a gated metric) is not measured here: there is no relation or
temporal question, so the gate reports it as not measured.

Usage:
  uv run python scripts/eval_video_smoke.py [--datasets bdd100k_clear bdd100k_night ...]
      [--limit N] [--manifest PATH] [--results-dir DIR] [--no-mlflow]
--limit N takes the first N clips of each clip set.
"""

import argparse
import json
import logging
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from discern.eval.datasets import DATA_DIR
from discern.eval.synth_clips import ClipEntry, Manifest
from discern.eval.video_metrics import box_f1_at_frames, gpu_seconds_per_video_second
from discern.eval.video_smoke import (
    ClipRecord,
    aggregate,
    count_question,
    ground_truth_frames,
    locate_label,
    locate_question,
    markdown_table,
    parse_count,
)
from discern.serve.engine import Engine, build_engine, free_gpu
from discern.video.io import decode_limit, iter_sampled_frames

logger = logging.getLogger("eval_video_smoke")
MLRUNS = Path(__file__).resolve().parents[1] / "mlruns"


def sampled_indices(engine: Engine, path: Path) -> list[int]:
    """Decode indices of the frames the pipeline samples (the keys of Track.frames)."""
    profile = engine.settings.profile
    return [
        f.index
        for f in iter_sampled_frames(
            path, profile.sample_fps, profile.max_long_side_px, decode_limit(engine.settings)
        )
    ]


def track_rows(engine: Engine, session_id: str) -> list[dict[str, Any]]:
    """Every track of every full scan in the conversation, accepted or rejected."""
    session = engine.session(session_id)
    rows: list[dict[str, Any]] = []
    for targets, tracks in session.conversation.scan_cache.items():
        for t in tracks:
            rows.append(
                {
                    "scan": "+".join(targets),
                    "id": t.id,
                    "label": t.label,
                    "status": t.status,
                    "t_start": t.t_start,
                    "t_end": t.t_end,
                    "n_frames": len(t.frames),
                    "mean_score": t.mean_score,
                    "rationale": t.rationale,
                }
            )
    return rows


def run_clip(engine: Engine, entry: ClipEntry, clips_dir: Path) -> ClipRecord:
    path = clips_dir / entry.clip
    rec = ClipRecord(entry.clip, entry.name, entry.image_id, entry.duration_seconds)
    rec.expected = dict(entry.expected_counts)
    session_id: str | None = None
    wall = 0.0
    try:
        started = time.perf_counter()
        session = engine.ingest(path)
        session_id = session.session_id
        wall += time.perf_counter() - started
        rec.duration = session.ingest.info.duration
        for label in sorted(entry.expected_counts):
            started = time.perf_counter()
            text, evidence = engine.ask(session, count_question(label))
            wall += time.perf_counter() - started
            rec.answers[label] = text
            rec.predicted[label] = parse_count(text)
            rec.fact_counts[label] = evidence.count
            print(
                f"  {label}: expected {entry.expected_counts[label]}, answer {text!r}", flush=True
            )
        rec.locate_label = locate_label(entry.expected_counts)
        if rec.locate_label is not None:
            started = time.perf_counter()
            text, evidence = engine.ask(session, locate_question(rec.locate_label))
            wall += time.perf_counter() - started
            rec.locate_answer = text
            result = (
                session.conversation.result_sets.get(evidence.result_set_id)
                if evidence.result_set_id
                else None
            )
            tracks = [
                t.model_copy(update={"label": rec.locate_label})
                for t in (result.facts.tracks if result else [])
            ]
            truth = ground_truth_frames(entry, rec.locate_label, sampled_indices(engine, path))
            rec.box_counts = box_f1_at_frames(tracks, truth)
        rec.tracks = track_rows(engine, session_id)
        rec.events = engine.trace_events(session_id)
        rec.wall_seconds_per_video_second = wall / rec.duration
        gpu = gpu_seconds_per_video_second(rec.events, rec.duration)
        rec.gpu_seconds_per_video_second = gpu if gpu > 0 else rec.wall_seconds_per_video_second
    except Exception as err:
        logger.error("clip %s failed:\n%s", entry.clip, traceback.format_exc())
        rec.error = f"{type(err).__name__}: {err}"
    finally:
        if session_id is not None:
            engine.store.delete(session_id)
            engine._sessions.pop(session_id, None)  # drop the in-memory index and tracks too
        free_gpu(None)
    return rec


def log_to_mlflow(
    experiment: str,
    run_name: str,
    metrics: dict[str, float],
    params: dict[str, str | int | float],
    table: str,
) -> None:
    import mlflow  # imported here so importing this module needs neither mlflow nor a database

    mlflow.set_tracking_uri(f"sqlite:///{(MLRUNS / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_text(table, "summary.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DATA_DIR / "clips" / "manifest.json")
    ap.add_argument("--datasets", nargs="*", help="clip sets to run (default: all in the manifest)")
    ap.add_argument("--limit", type=int, default=0, help="clips per clip set (0 = all)")
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--profile", default=None, help="profile name (default: DISCERN_PROFILE)")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    clips_dir = args.manifest.parent
    results_dir = args.results_dir or clips_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    by_set: dict[str, list[ClipEntry]] = defaultdict(list)
    for entry in manifest.clips:
        if not args.datasets or entry.name in args.datasets:
            by_set[entry.name].append(entry)
    if not by_set:
        raise SystemExit("no clips selected; check --datasets and the manifest")

    engine = build_engine(args.profile)
    settings = engine.settings
    records: dict[str, list[ClipRecord]] = {}
    for name, entries in by_set.items():
        records[name] = []
        for entry in entries[: args.limit or None]:
            print(f"[{name}] {entry.clip}", flush=True)
            rec = run_clip(engine, entry, clips_dir)
            records[name].append(rec)
            out = results_dir / f"{name}__{entry.image_id}.json"
            out.write_text(json.dumps(rec.to_json_dict(), indent=1), encoding="utf-8")

    metrics = {name: aggregate(recs) for name, recs in records.items()}
    every = [r for recs in records.values() for r in recs]
    metrics["all"] = aggregate(every)
    table = markdown_table(metrics)
    (results_dir / "summary.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print("\n" + table)

    if not args.no_mlflow:
        experiment = settings.thresholds.eval_gate.experiment
        common: dict[str, str | int | float] = {
            "profile": settings.profile.name,
            "config_hash": settings.config_hash,
            "models": json.dumps(settings.profile.models, sort_keys=True),
            "sample_fps": settings.profile.sample_fps,
            "manifest": args.manifest.name,
        }
        for name, recs in records.items():  # the overall run goes last: the gate reads the latest
            params = {**common, "dataset": name, "n_clips": len(recs)}
            log_to_mlflow(experiment, f"video-smoke/{name}", metrics[name], params, table)
        params = {**common, "dataset": ",".join(records), "n_clips": len(every)}
        log_to_mlflow(experiment, "video-smoke/all", metrics["all"], params, table)
        print(f"logged to MLflow experiment {experiment!r}")


if __name__ == "__main__":
    main()
