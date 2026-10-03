from pathlib import Path

from discern.config.settings import EvalGateThresholds
from discern.experience.promotion import (
    promote_candidate,
    promote_if_passes,
    read_pointer,
    write_pointer,
)

GATE = EvalGateThresholds(
    experiment="x",
    tolerance=0.1,
    minimums={"box_f1": 0.5},
    maximums={"gpu_seconds_per_video_second": 2.0},
)
BASE = {"box_f1": 0.6, "gpu_seconds_per_video_second": 1.0}


def test_pass_when_gate_and_baseline_hold() -> None:
    d = promote_if_passes({"box_f1": 0.58, "gpu_seconds_per_video_second": 1.05}, BASE, GATE)
    assert d.promote and d.reasons == ()


def test_fails_the_gate() -> None:
    d = promote_if_passes({"box_f1": 0.4, "gpu_seconds_per_video_second": 1.0}, {}, GATE)
    assert not d.promote and "box_f1" in d.reasons[0]


def test_fails_on_regression_against_baseline_even_if_gate_passes() -> None:
    d = promote_if_passes({"box_f1": 0.52, "gpu_seconds_per_video_second": 1.0}, BASE, GATE)
    assert not d.promote  # 0.52 passes the gate (floor 0.45) but is below 0.6 * 0.9 = 0.54
    assert "regresses" in d.reasons[0]


def test_cost_regression_fails() -> None:
    d = promote_if_passes({"box_f1": 0.6, "gpu_seconds_per_video_second": 1.2}, BASE, GATE)
    assert not d.promote


def test_missing_candidate_metric_fails() -> None:
    assert not promote_if_passes({"box_f1": 0.6}, BASE, GATE).promote


def test_pointer_written_only_on_pass(tmp_path: Path) -> None:
    pointer = tmp_path / "memory" / "pinned.json"
    assert read_pointer(pointer) is None
    write_pointer(pointer, "v1")
    bad = {"box_f1": 0.1, "gpu_seconds_per_video_second": 1.0}
    assert not promote_candidate("v2", bad, BASE, GATE, pointer).promote
    assert read_pointer(pointer) == "v1"
    good = {"box_f1": 0.65, "gpu_seconds_per_video_second": 1.0}
    assert promote_candidate("v2", good, BASE, GATE, pointer).promote
    assert read_pointer(pointer) == "v2"
