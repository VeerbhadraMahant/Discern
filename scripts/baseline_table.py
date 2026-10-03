"""Print the baseline results table (F1@0.5 on the gate subset) from the MLflow runs."""

from pathlib import Path

import mlflow

from discern.eval.datasets import DATA_DIR  # noqa: F401  (keeps import path identical to runners)

MLRUNS = Path(__file__).resolve().parents[1] / "mlruns"
mlflow.set_tracking_uri(f"sqlite:///{(MLRUNS / 'mlflow.db').as_posix()}")

runs = mlflow.search_runs(experiment_names=["m1-baselines"], order_by=["start_time ASC"])
latest: dict[tuple[str, str], float] = {}
for _, r in runs.iterrows():  # later runs overwrite earlier ones for the same (method, dataset)
    latest[(r["params.method"], r["params.dataset"])] = r["metrics.f1"]

datasets = sorted({d for _, d in latest})
methods = sorted({m for m, _ in latest})
print("| method | " + " | ".join(datasets) + " |")
print("|-|" + "-|" * len(datasets))
for m in methods:
    cells = [f"{latest[(m, d)]:.3f}" if (m, d) in latest else "n/a" for d in datasets]
    print(f"| {m} | " + " | ".join(cells) + " |")
