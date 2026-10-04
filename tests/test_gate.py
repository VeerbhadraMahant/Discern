from discern.config import load_settings
from discern.config.settings import EvalGateThresholds
from discern.eval.gate import evaluate_gate

GATE = EvalGateThresholds(
    experiment="gate",
    tolerance=0.1,
    minimums={"count_accuracy": 0.8, "box_f1": 0.5},
    maximums={"count_mae": 1.0},
)
GOOD = {"count_accuracy": 0.85, "box_f1": 0.6, "count_mae": 0.5}


def test_passes_when_every_metric_is_within_bounds() -> None:
    result = evaluate_gate(GOOD, GATE)
    assert result.passed and result.reasons == ()


def test_tolerance_allows_a_small_drop() -> None:
    # 0.8 * (1 - 0.1) = 0.72
    assert evaluate_gate({**GOOD, "count_accuracy": 0.73}, GATE).passed
    assert not evaluate_gate({**GOOD, "count_accuracy": 0.71}, GATE).passed


def test_deliberately_broken_change_fails_the_gate() -> None:
    broken = {"count_accuracy": 0.2, "box_f1": 0.0, "count_mae": 4.0}
    result = evaluate_gate(broken, GATE)
    assert not result.passed
    assert len(result.reasons) == 3
    assert any("count_accuracy" in r and "below" in r for r in result.reasons)
    assert any("count_mae" in r and "above" in r for r in result.reasons)


def test_maximum_has_tolerance_too() -> None:
    assert evaluate_gate({**GOOD, "count_mae": 1.1}, GATE).passed
    assert not evaluate_gate({**GOOD, "count_mae": 1.2}, GATE).passed


def test_missing_and_nan_metrics_fail() -> None:
    missing = evaluate_gate({"count_accuracy": 0.9, "count_mae": 0.1}, GATE)
    assert not missing.passed and "box_f1" in missing.reasons[0]
    assert not evaluate_gate({**GOOD, "box_f1": float("nan")}, GATE).passed


def test_extra_metrics_are_ignored() -> None:
    assert evaluate_gate({**GOOD, "unrelated": -5.0}, GATE).passed


def test_configured_thresholds_are_set_and_a_sane_result_passes() -> None:
    gate = load_settings("local_lite").thresholds.eval_gate
    assert gate.minimums and gate.maximums and gate.tolerance >= 0
    # Thresholds come from measured baselines: a result at the thresholds themselves passes.
    sane = dict(gate.minimums) | dict(gate.maximums)
    assert evaluate_gate(sane, gate).passed
    assert not evaluate_gate({}, gate).passed
