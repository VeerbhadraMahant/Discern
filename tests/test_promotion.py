import importlib.util
import json
from pathlib import Path

import pytest

import discern.config
import discern.feedback.schema
from discern.config.settings import EvalGateThresholds, load_settings
from discern.experience.promotion import (
    promote_candidate,
    promote_if_passes,
    read_pointer,
    write_pointer,
)
from discern.experience.schema import ExperienceRecord, ExperienceStore

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


def test_feedback_loop_pins_in_the_pointer_file_the_config_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "feedback_loop", root / "scripts/feedback_loop.py"
    )
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    settings = load_settings()
    serve = settings.thresholds.serve.model_copy(update={"memory_pointer": "live.json"})
    thresholds = settings.thresholds.model_copy(update={"serve": serve})
    monkeypatch.setattr(
        discern.config,
        "load_settings",
        lambda: settings.model_copy(update={"thresholds": thresholds}),
    )
    monkeypatch.setattr(discern.feedback.schema, "feedback_to_labelled_samples", lambda *a: [1])
    monkeypatch.setattr(script, "_harvest", lambda *a: None)
    record = ExperienceRecord(
        profile_key="fog|dim|poor|medium|dense",
        query_type="detect",
        node="sr",
        option="off",
        metric_name="f1_best",
        metric_value=0.5,
        sample_id="x",
        source="feedback",
        memory_version="v2",
    )
    ExperienceStore(tmp_path / "records.jsonl").append([record])
    gate = settings.thresholds.eval_gate
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({**gate.minimums, **gate.maximums}))
    args = [
        "--feedback-dir",
        str(tmp_path),
        "--version",
        "v2",
        "--metrics",
        str(metrics),
        "--baseline",
        str(metrics),
        "--memory-dir",
        str(tmp_path),
    ]

    assert script.main(args) == 0
    assert read_pointer(tmp_path / "live.json") == "v2"
    assert not (tmp_path / "pinned.json").exists()
