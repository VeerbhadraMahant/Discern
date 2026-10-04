import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from discern.agent.med import detect_image
from discern.agent.sair import plan_image
from discern.agent.schemas import DetectorInfo, SceneProfile
from discern.config.settings import load_settings
from discern.experience.aggregate import (
    ConfigStat,
    Memory,
    MemoryVersion,
    OptionStat,
    aggregate_configurations,
    build_memory,
    load_memory,
    with_configuration_stats,
    write_memory,
)
from discern.experience.policy import (
    ExperiencePolicy,
    JointExperiencePolicy,
    build_policy,
    detector_decision,
)
from discern.experience.schema import ExperienceRecord, Node
from discern.models.fakes import FakeDetector, FakeRestorer, FakeVLM
from discern.models.roles import Detection
from discern.trace import TraceCollector
from discern.vision.boxes import Box

SETTINGS = load_settings()
EXP = SETTINGS.thresholds.experience
MIN_N = EXP.policy_min_count
MARGIN = EXP.policy_margin
K = SETTINGS.thresholds.agent.top_k_detectors
PRIORITY = ["a", "b", "c"]
DEFAULT = "none|off|a+b"  # restorer none, SR off, the first K of PRIORITY (sorted)
BEST = "dehaze|auto|a+c"


def profile(label: str = "fog", visibility: str = "poor") -> SceneProfile:
    return SceneProfile.model_validate(
        {
            "scene_label": label,
            "illumination": "normal",
            "visibility": visibility,
            "object_scale": "small",
            "object_density": "dense",
            "confidence": 0.8,
        }
    )


def make_memory(
    cells: dict[str, tuple[float, int]],
    nodes: tuple[tuple[Node, str, float, int], ...] = (),
    key: str | None = None,
    extra: dict[str, dict[str, tuple[float, int]]] | None = None,
) -> Memory:
    """A memory with configuration stats (label -> (mean, count)) for one profile key, plus
    optional node-level stats and configuration stats for other profile keys."""
    key = key or profile().key
    per_key = {key: cells, **(extra or {})}
    configs = tuple(
        ConfigStat(
            profile_key=k, query_type="detect", configuration=label, mean=m, std=0.0, count=n
        )
        for k, rows in per_key.items()
        for label, (m, n) in rows.items()
    )
    stats = tuple(
        OptionStat(
            profile_key=key, query_type="detect", node=n, option=o, mean=m, std=0.0, count=c
        )
        for n, o, m, c in nodes
    )
    version = MemoryVersion(
        id="v1", created=datetime(2026, 1, 1, tzinfo=UTC), record_count=1, source_hash="h"
    )
    return Memory(version=version, stats=stats, config_stats=configs)


def joint(mem: Memory, label: str = "fog", priority: list[str] = PRIORITY) -> JointExperiencePolicy:
    return JointExperiencePolicy.from_memory(mem, profile(label), EXP, priority, K)


# The interaction case: each node on its own prefers none / off / a+b (the marginal winners are
# the default's parts), but the best whole configuration is dehaze + SR + a+c.
INTERACTION_CELLS = {
    DEFAULT: (0.50, 50),
    "none|off|a+c": (0.40, 50),
    "none|auto|a+b": (0.30, 50),
    "dehaze|off|a+b": (0.30, 50),
    BEST: (0.70, 50),
}
INTERACTION_NODES: tuple[tuple[Node, str, float, int], ...] = (
    ("restorer", "none", 0.76, 50),
    ("restorer", "dehaze", 0.72, 50),
    ("sr", "off", 0.76, 50),
    ("sr", "auto", 0.72, 50),
    ("detector_set", "a+b", 0.76, 50),
    ("detector_set", "a+c", 0.72, 50),
)


def test_joint_picks_the_best_whole_configuration_where_nodewise_decides_differently() -> None:
    mem = make_memory(INTERACTION_CELLS, INTERACTION_NODES)
    node_wise = ExperiencePolicy.from_memory(mem, profile(), EXP)
    assert [node_wise.decide(n) for n in ("restorer", "sr", "detector_set")] == [
        "none",
        "off",
        "a+b",
    ]
    p = joint(mem)
    chosen = [p.decision(n) for n in ("restorer", "sr", "detector_set")]
    assert [d.option for d in chosen if d] == ["mapped", "auto", "a+c"]
    found = p.decision("sr")
    assert found is not None
    assert found.rationale == (
        "experience joint policy: dehaze|auto|a+c 0.700 vs default 0.500, n=50"
    )
    assert p.mapped_restorer == "dehaze"


def test_margin_gate_below_none_and_exactly_the_margin_decides() -> None:
    below = make_memory({DEFAULT: (0.50, 50), BEST: (0.50 + MARGIN - 0.005, 50)})
    assert joint(below).decision("sr") is None
    exact = make_memory({DEFAULT: (0.50, 50), BEST: (0.50 + MARGIN, 50)})
    assert joint(exact).decision("sr") is not None


def test_count_gate_applies_to_the_winner_and_to_the_default() -> None:
    few_winner = make_memory({DEFAULT: (0.50, 50), BEST: (0.90, MIN_N - 1)})
    assert joint(few_winner).decision("restorer") is None  # ignored, nothing else beats default
    ok_next = make_memory(
        {DEFAULT: (0.50, 50), BEST: (0.90, MIN_N - 1), "dehaze|off|a+b": (0.70, 50)}
    )
    found = joint(ok_next).decision("restorer")
    assert found is not None and "dehaze|off|a+b 0.700" in found.rationale
    few_default = make_memory({DEFAULT: (0.50, MIN_N - 1), BEST: (0.90, 50)})
    assert joint(few_default).decision("restorer") is None
    assert joint(make_memory({BEST: (0.90, 50)})).decision("restorer") is None  # no default row


def test_the_comparison_is_against_the_default_configuration_not_the_runner_up() -> None:
    # The runner-up is within the margin of the winner but both clearly beat the default.
    mem = make_memory({DEFAULT: (0.40, 50), BEST: (0.70, 50), "none|auto|a+c": (0.69, 50)})
    found = joint(mem).decision("detector_set")
    assert found is not None and found.rationale.endswith("0.700 vs default 0.400, n=50")


def test_a_winner_that_is_the_default_decides_nothing() -> None:
    mem = make_memory({DEFAULT: (0.90, 50), BEST: (0.50, 50)})
    assert joint(mem).decision("restorer") is None


def test_the_default_detector_pair_follows_the_priority_order() -> None:
    mem = make_memory(
        {"none|off|b+c": (0.40, 50), "dehaze|off|b+c": (0.70, 50), DEFAULT: (0.8, 50)}
    )
    assert joint(mem, priority=["c", "b", "a"]).decision("restorer") is not None  # default b+c
    assert joint(mem, priority=["a", "b", "c"]).decision("restorer") is None  # default a+b wins


def test_only_k_distinct_available_detectors_and_known_options_are_candidates() -> None:
    cells = {
        DEFAULT: (0.50, 50),
        "dehaze|off|a": (0.99, 50),  # one detector, not K
        "dehaze|off|a+z": (0.99, 50),  # z is not available
        "derain|off|a+b": (0.99, 50),  # not this scene's restorer
        "dehaze|x4|a+b": (0.99, 50),  # not a harvested SR option
        "dehaze|off": (0.99, 50),  # malformed
    }
    assert joint(make_memory(cells)).decision("restorer") is None


def test_detector_decision_accept_filter_and_helper() -> None:
    mem = make_memory({DEFAULT: (0.50, 50), BEST: (0.70, 50)})
    p = joint(mem)
    assert p.decision("detector_set", lambda o: False) is None
    found = detector_decision(p, K, ["a", "b", "c"])
    assert found is not None and found.detectors == ["a", "c"]
    assert detector_decision(p, K, ["a", "b"]) is None  # c unavailable at detection time
    assert detector_decision(p, K + 1, ["a", "b", "c"]) is None


def test_scene_without_a_mapped_restorer_still_decides_sr_and_detectors() -> None:
    cells = {DEFAULT: (0.50, 50), "none|auto|a+c": (0.70, 50)}
    p = joint(make_memory(cells, key=profile("normal").key), label="normal")
    restorer, sr, detectors = (p.decision(n) for n in ("restorer", "sr", "detector_set"))
    assert restorer is not None and restorer.option == "none"
    assert sr is not None and sr.option == "auto"
    assert detectors is not None and detectors.option == "a+c"


def test_configuration_stats_are_pooled_over_similar_profiles_weighted_by_count() -> None:
    near = profile(visibility="moderate").key
    mem = make_memory(
        {DEFAULT: (0.40, 30), BEST: (0.60, 30)},
        extra={near: {DEFAULT: (0.40, 10), BEST: (0.80, 10)}},
    )
    found = joint(mem).decision("restorer")
    # BEST pools to (0.6 * 30 + 0.8 * 10) / 40 = 0.65, n = 40
    assert found is not None and found.rationale.endswith("0.650 vs default 0.400, n=40")


def test_no_config_stats_or_other_scene_decides_nothing() -> None:
    assert joint(make_memory({})).decision("restorer") is None
    mem = make_memory({DEFAULT: (0.5, 50), BEST: (0.9, 50)})
    assert joint(mem, label="rain").decision("restorer") is None


# ---- build_policy -----------------------------------------------------------------------------


def test_build_policy_by_mode_and_fallbacks() -> None:
    mem = make_memory({DEFAULT: (0.50, 50), BEST: (0.70, 50)}, nodes=(("sr", "off", 0.5, 50),))
    node_settings = EXP.model_copy(update={"policy_mode": "node"})
    joint_settings = EXP.model_copy(update={"policy_mode": "joint"})
    assert isinstance(build_policy(mem, profile(), node_settings, PRIORITY, K), ExperiencePolicy)
    assert isinstance(
        build_policy(mem, profile(), joint_settings, PRIORITY, K), JointExperiencePolicy
    )
    for settings in (node_settings, joint_settings):
        assert build_policy(None, profile(), settings, PRIORITY, K) is None
    old = mem.model_copy(update={"config_stats": ()})  # a memory without configuration stats
    assert build_policy(old, profile(), joint_settings, PRIORITY, K) is None
    assert isinstance(build_policy(old, profile(), node_settings, PRIORITY, K), ExperiencePolicy)


def test_default_policy_mode_is_joint() -> None:
    assert EXP.policy_mode == "joint"


# ---- the joint decision applied by plan_image and detect_image ----------------------------------


def test_plan_image_applies_the_joint_decision_with_one_rationale_and_no_extra_vlm_calls() -> None:
    mem = make_memory({DEFAULT: (0.50, 50), BEST: (0.70, 50)})
    vlm = FakeVLM([profile().model_dump_json()])
    trace = TraceCollector()
    image = np.full((64, 96, 3), 100, dtype=np.uint8)
    plan, out = plan_image(
        vlm,
        trace,
        image,
        {"dehaze": FakeRestorer("dehaze", offset=20)},
        policy=lambda _: joint(mem),
    )
    assert (plan.restorer, plan.use_restored, plan.sr_factor) == ("dehaze", True, 4)
    assert int(out[0, 0, 0]) == 120 and len(vlm.prompts) == 1
    rationale = "experience joint policy: dehaze|auto|a+c 0.700 vs default 0.500, n=50"
    assert plan.decisions.count(rationale) == 1
    assert [e.rationale for e in trace.events if e.node.startswith("experience_policy")] == [
        rationale,
        rationale,
    ]


def test_detect_image_runs_only_the_joint_detector_set() -> None:
    def car(detector: str, x: float) -> Detection:
        return Detection(box=Box(x, 10, x + 30, 40), label="car", score=0.9, detector=detector)

    detectors = {n: FakeDetector(n, [car(n, 10 + 90 * i)]) for i, n in enumerate(PRIORITY)}
    catalog = [DetectorInfo(name=n, capabilities="x", speed_class="fast") for n in PRIORITY]
    mem = make_memory({DEFAULT: (0.50, 50), BEST: (0.70, 50)})
    decision = detector_decision(joint(mem), K, PRIORITY)
    assert decision is not None
    vlm = FakeVLM([])
    found = detect_image(
        vlm, TraceCollector(), np.zeros((60, 300, 3), np.uint8), ["car"], profile(), detectors,
        catalog, adjudicate=False, settings=SETTINGS, priority=PRIORITY,
        preferred=decision.detectors,
    )
    assert vlm.prompts == [] and sorted(d.detector for d in found) == ["a", "c"]


# ---- configuration stats in the memory file ---------------------------------------------------


def record(node: str, option: str, value: float, sample: str, metric: str) -> ExperienceRecord:
    return ExperienceRecord.model_validate(
        {
            "profile_key": profile().key,
            "query_type": "detect",
            "node": node,
            "option": option,
            "metric_name": metric,
            "metric_value": value,
            "sample_id": sample,
            "source": "benchmark",
            "memory_version": "v1",
        }
    )


RECORDS = [
    record("sr", "off", 0.5, "s1", "f1_best"),
    record("configuration", DEFAULT, 0.4, "s1", "f1_fused"),
    record("configuration", DEFAULT, 0.6, "s2", "f1_fused"),
    record("configuration", BEST, 0.8, "s1", "f1_adjudicated"),
]


def test_aggregate_configurations_gives_mean_std_and_count_per_label() -> None:
    stats = {c.configuration: c for c in aggregate_configurations(RECORDS)}
    assert set(stats) == {DEFAULT, BEST}
    assert stats[DEFAULT].mean == pytest.approx(0.5)
    assert stats[DEFAULT].std == pytest.approx(0.1)
    assert stats[DEFAULT].count == 2 and stats[BEST].count == 1


def test_build_memory_keeps_node_stats_and_adds_configuration_stats(tmp_path: Path) -> None:
    mem = build_memory(RECORDS, "v1", datetime(2026, 1, 1, tzinfo=UTC))
    assert [s.option for s in mem.stats] == ["off"]  # configuration rows stay out of node stats
    assert len(mem.config_stats) == 2
    path = write_memory(mem, tmp_path)
    assert load_memory(path) == mem


def test_a_memory_file_without_configuration_stats_still_loads(tmp_path: Path) -> None:
    mem = build_memory(RECORDS, "v1", datetime(2026, 1, 1, tzinfo=UTC))
    path = write_memory(mem, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["config_stats"]  # the format written before configuration stats existed
    path.write_text(json.dumps(data), encoding="utf-8")
    old = load_memory(path)
    assert old.config_stats == () and old.stats == mem.stats and old.version == mem.version
    rebuilt = with_configuration_stats(old, RECORDS)
    assert rebuilt == mem


# ---- engine -----------------------------------------------------------------------------------


def test_engine_policy_falls_back_without_memory_or_configuration_stats(tmp_path: Path) -> None:
    from tests.test_serve_engine import make_engine

    engine = make_engine(tmp_path, FakeVLM([]))
    assert engine._policy(profile()) is None  # no memory
    names = engine._available_detectors()
    engine.memory = make_memory({}).model_copy(update={"config_stats": ()})
    assert engine._policy(profile()) is None  # joint mode, memory without configuration stats
    pair = "+".join(sorted(names[:K]))
    engine.memory = make_memory(
        {f"none|off|{pair}": (0.4, 50), f"dehaze|off|{pair}": (0.7, 50)}
    )
    found = engine._policy(profile())
    assert isinstance(found, JointExperiencePolicy)
    decision = found.decision("restorer")
    assert decision is not None and decision.option == "mapped"
