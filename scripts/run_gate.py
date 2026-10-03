"""Evaluation gate: read the latest finished gate run from MLflow and fail on regressions.

Usage: uv run python scripts/run_gate.py
Exit code 0 when every gated metric is within its threshold (configs/thresholds.yaml,
`eval_gate`), 1 when any is not, 2 when there is no run to check.
The gate runs on a self-hosted GPU runner after the gate subset has been evaluated and logged
to the experiment named in `eval_gate.experiment`.
"""

import sys
from pathlib import Path

from discern.config import load_settings
from discern.eval.gate import evaluate_gate

MLRUNS = Path(__file__).resolve().parents[1] / "mlruns"
TRACKING_URI = f"sqlite:///{(MLRUNS / 'mlflow.db').as_posix()}"


def main() -> int:
    import mlflow  # imported here so importing this module needs neither mlflow nor a database

    gate = load_settings().thresholds.eval_gate
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(
        experiment_names=[gate.experiment],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        print(f"gate: no finished run in experiment {gate.experiment!r}", file=sys.stderr)
        return 2
    run = runs[0]
    result = evaluate_gate(run.data.metrics, gate)
    print(f"gate: run {run.info.run_id} ({run.info.run_name})")
    if result.passed:
        print("gate: PASSED")
        return 0
    print("gate: FAILED", file=sys.stderr)
    for reason in result.reasons:
        print(f"  - {reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
