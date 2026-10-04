"""Log one combined video-smoke run from the latest per-dataset runs in the gate experiment.

Each metric is the mean of the per-dataset values weighted by the number of clips, so it is an
approximation of one run over all clips (pooled metrics such as box F1 and the verifier pass rate
are not recomputed from raw counts). The run is tagged `combined` so nobody mistakes it for a
single end-to-end run. Usage: uv run python scripts/combine_video_runs.py
"""

from pathlib import Path

import mlflow

from discern.config import load_settings

DATASETS = ["bdd100k_clear", "bdd100k_night", "bdd100k_rainy", "bdd100k_clear_fog"]
SUMMED = {"clips_ok", "clips_failed", "vlm_fallback_events"}


def main() -> None:
    settings = load_settings()
    experiment = settings.thresholds.eval_gate.experiment
    mlruns = Path(__file__).resolve().parents[1] / "mlruns"
    mlflow.set_tracking_uri(f"sqlite:///{(mlruns / 'mlflow.db').as_posix()}")
    runs = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time ASC"])
    mlflow.set_experiment(experiment)
    latest = {}
    for _, row in runs.iterrows():  # later runs overwrite earlier ones per dataset
        name = str(row.get("tags.mlflow.runName", ""))
        if name.startswith("video-smoke/") and name.split("/", 1)[1] in DATASETS:
            latest[name.split("/", 1)[1]] = row
    missing = [d for d in DATASETS if d not in latest]
    if missing:
        raise SystemExit(f"no per-dataset run for: {missing}")

    weights = {d: float(latest[d]["params.n_clips"]) for d in DATASETS}
    total = sum(weights.values())
    names = sorted(
        {
            c.removeprefix("metrics.")
            for d in DATASETS
            for c in latest[d].index
            if c.startswith("metrics.")
        }
    )
    metrics: dict[str, float] = {}
    for name in names:
        values = {d: latest[d].get(f"metrics.{name}") for d in DATASETS}
        values = {d: v for d, v in values.items() if v == v and v is not None}  # drop NaN
        if not values:
            continue
        if name in SUMMED:
            metrics[name] = float(sum(values.values()))
        else:
            w = sum(weights[d] for d in values)
            metrics[name] = float(sum(v * weights[d] for d, v in values.items()) / w)
    with mlflow.start_run(run_name="video-smoke/all"):
        mlflow.log_params(
            {
                "combined": "clip-weighted mean of the latest per-dataset runs",
                "datasets": ",".join(DATASETS),
                "n_clips": int(total),
                "config_hash": settings.config_hash,
            }
        )
        mlflow.log_metrics(metrics)
    for k, v in sorted(metrics.items()):
        print(f"{k:32s} {v:.3f}")


if __name__ == "__main__":
    main()
