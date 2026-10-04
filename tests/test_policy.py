from datetime import UTC, datetime

import numpy as np
import pytest

from discern.agent.med import detect_image
from discern.agent.nodes.detector_select import detector_select
from discern.agent.sair import plan_image
from discern.agent.schemas import DetectorInfo, SceneProfile
from discern.config.settings import load_settings
from discern.experience.aggregate import Memory, MemoryVersion, OptionStat
from discern.experience.policy import ExperiencePolicy, detector_decision
from discern.experience.schema import Node
from discern.models.fakes import FakeDetector, FakeRestorer, FakeVLM
from discern.models.roles import Detection, Image
from discern.trace import TraceCollector
from discern.vision.boxes import Box

SETTINGS = load_settings()
EXP = SETTINGS.thresholds.experience
MIN_N = EXP.policy_min_count
MARGIN = EXP.policy_margin
TARGET = SETTINGS.thresholds.agent.sr_target_long_side
OPTION_ORDER: tuple[Node, ...] = ("restorer", "sr", "detector_set")


def profile(label: str = "fog") -> SceneProfile:
    return SceneProfile.model_validate(
        {
            "scene_label": label,
            "illumination": "normal",
            "visibility": "poor",
            "object_scale": "small",
            "object_density": "dense",
            "confidence": 0.8,
        }
    )


def memory(*cells: tuple[Node, str, float, int], label: str = "fog") -> Memory:
    key = profile(label).key
    stats = tuple(
        OptionStat(
            profile_key=key, query_type="detect", node=n, option=o, mean=m, std=0.0, count=c
        )
        for n, o, m, c in cells
    )
    version = MemoryVersion(
        id="v1", created=datetime(2026, 1, 1, tzinfo=UTC), record_count=len(stats), source_hash="h"
    )
    return Memory(version=version, stats=stats)


def policy(mem: Memory, label: str = "fog") -> ExperiencePolicy:
    return ExperiencePolicy.from_memory(mem, profile(label), EXP)


STRONG = memory(
    ("restorer", "none", 0.76, 50),
    ("restorer", "dehaze", 0.60, 50),
    ("sr", "off", 0.76, 50),
    ("sr", "auto", 0.695, 50),
    ("detector_set", "a+b", 0.80, 50),
    ("detector_set", "a+c", 0.70, 50),
    ("detector_set", "b+c", 0.50, 50),
)


# ---- ExperiencePolicy -------------------------------------------------------------------------


def test_strong_evidence_decides_every_node_with_numbers_in_the_rationale() -> None:
    p = policy(STRONG)
    assert p.decide("restorer") == "none"
    assert p.decide("sr") == "off"
    assert p.decide("detector_set") == "a+b"
    sr = p.decision("sr")
    assert sr is not None and sr.rationale == "experience policy: sr off (0.760 vs 0.695, n=50)"
    restorer = p.decision("restorer")
    assert restorer is not None and "restorer none (0.760 vs 0.600, n=50)" in restorer.rationale


def test_mapped_restorer_and_auto_sr_can_win() -> None:
    p = policy(
        memory(
            ("restorer", "none", 0.40, MIN_N),
            ("restorer", "dehaze", 0.60, MIN_N),
            ("sr", "off", 0.40, MIN_N),
            ("sr", "auto", 0.60, MIN_N),
        )
    )
    assert p.decide("restorer") == "mapped" and p.mapped_restorer == "dehaze"
    assert p.decide("sr") == "auto"
    decision = p.decision("restorer")
    assert decision is not None and "restorer dehaze" in decision.rationale


def test_margin_below_the_setting_returns_none_and_exactly_the_margin_decides() -> None:
    below = memory(("sr", "off", 0.60 + MARGIN / 2, 50), ("sr", "auto", 0.60, 50))
    assert policy(below).decide("sr") is None
    exact = memory(("sr", "off", round(0.60 + MARGIN, 6), 50), ("sr", "auto", 0.60, 50))
    assert policy(exact).decide("sr") == "off"


def test_too_few_samples_for_either_option_returns_none() -> None:
    few = memory(("sr", "off", 0.9, MIN_N - 1), ("sr", "auto", 0.1, 50))
    assert policy(few).decide("sr") is None
    few_other = memory(("sr", "off", 0.9, 50), ("sr", "auto", 0.1, MIN_N - 1))
    assert policy(few_other).decide("sr") is None
    enough = memory(("sr", "off", 0.9, MIN_N), ("sr", "auto", 0.1, MIN_N))
    assert policy(enough).decide("sr") == "off"


def test_ties_and_a_missing_option_return_none() -> None:
    assert policy(memory(("sr", "off", 0.5, 50), ("sr", "auto", 0.5, 50))).decide("sr") is None
    assert policy(memory(("sr", "off", 0.9, 50))).decide("sr") is None


def test_empty_memory_and_unknown_profile_decide_nothing() -> None:
    for p in (policy(memory()), policy(STRONG, label="underwater")):
        assert [p.decide(n) for n in OPTION_ORDER] == [None, None, None]


def test_a_scene_without_a_mapped_restorer_never_decides_the_restorer() -> None:
    mem = memory(("restorer", "none", 0.9, 50), ("restorer", "x", 0.1, 50), label="normal")
    assert policy(mem, label="normal").decide("restorer") is None


def test_detector_options_below_min_count_are_ignored_not_ranked() -> None:
    mem = memory(
        ("detector_set", "a+b", 0.95, MIN_N - 1),
        ("detector_set", "a+c", 0.80, 50),
        ("detector_set", "b+c", 0.60, 50),
    )
    assert policy(mem).decide("detector_set") == "a+c"


def test_detector_decision_needs_exactly_k_available_detectors() -> None:
    k = SETTINGS.thresholds.agent.top_k_detectors
    p = policy(STRONG)
    found = detector_decision(p, k, ["a", "b", "c"])
    assert found is not None and found.detectors == ["a", "b"]
    assert detector_decision(p, k, ["a", "c"]) is None  # one usable option: nothing to compare
    assert detector_decision(p, k, ["a"]) is None
    assert detector_decision(None, k, ["a", "b"]) is None
    singles = policy(memory(("detector_set", "a", 0.9, 50), ("detector_set", "b", 0.1, 50)))
    assert detector_decision(singles, k, ["a", "b"]) is None


# ---- plan_image with a policy -----------------------------------------------------------------


def small_image() -> Image:
    return np.full((64, 96, 3), 100, dtype=np.uint8)


def restorers() -> dict[str, FakeRestorer]:
    return {"dehaze": FakeRestorer("dehaze", offset=20)}


def perception_only() -> FakeVLM:
    return FakeVLM([profile().model_dump_json()])


def test_policy_none_and_policy_returning_none_keep_every_vlm_call() -> None:
    script = [
        profile().model_dump_json(),
        '{"restorer": "dehaze"}',
        '{"choice": "restored"}',
        '{"factor": 4}',
    ]
    for pol in (None, lambda _: None, policy(memory())):
        vlm = FakeVLM(script)
        plan, _ = plan_image(vlm, TraceCollector(), small_image(), restorers(), policy=pol)
        assert len(vlm.prompts) == 4
        assert (plan.restorer, plan.use_restored, plan.sr_factor) == ("dehaze", True, 4)


def test_policy_none_restorer_and_sr_off_skip_all_vlm_calls_but_perception() -> None:
    trace = TraceCollector()
    vlm, img = perception_only(), small_image()
    plan, out = plan_image(vlm, trace, img, restorers(), policy=policy(STRONG))
    assert (plan.restorer, plan.use_restored, plan.sr_factor) == ("none", False, None)
    assert out is img and len(vlm.prompts) == 1
    assert "experience policy: sr off (0.760 vs 0.695, n=50)" in plan.decisions
    assert [e.node for e in trace.events] == [
        "perception",
        "experience_policy.restorer",
        "experience_policy.sr",
    ]
    assert trace.events[2].rationale == "experience policy: sr off (0.760 vs 0.695, n=50)"


def test_policy_mapped_restorer_skips_image_select_and_sr_auto_uses_the_factor() -> None:
    mem = memory(
        ("restorer", "none", 0.4, 50),
        ("restorer", "dehaze", 0.6, 50),
        ("sr", "off", 0.4, 50),
        ("sr", "auto", 0.6, 50),
    )
    vlm = perception_only()
    plan, out = plan_image(
        vlm, TraceCollector(), small_image(), restorers(), policy=lambda p: policy(mem)
    )
    assert (plan.restorer, plan.use_restored, plan.sr_factor) == ("dehaze", True, 4)
    assert int(out[0, 0, 0]) == 120 and len(vlm.prompts) == 1
    assert any(d.startswith("experience policy: restorer dehaze") for d in plan.decisions)


def test_policy_sr_auto_is_off_when_the_image_already_reaches_the_target() -> None:
    mem = memory(("sr", "off", 0.4, 50), ("sr", "auto", 0.6, 50))
    wide = np.zeros((8, TARGET, 3), dtype=np.uint8)
    vlm = FakeVLM([profile().model_dump_json(), '{"restorer": "none"}'])
    plan, _ = plan_image(vlm, TraceCollector(), wide, restorers(), policy=policy(mem))
    assert plan.sr_factor is None and len(vlm.prompts) == 2  # perception and restorer_select


def test_policy_mapped_restorer_that_is_unavailable_falls_back_to_none() -> None:
    mem = memory(("restorer", "none", 0.4, 50), ("restorer", "dehaze", 0.6, 50))
    vlm = FakeVLM([profile().model_dump_json(), '{"factor": "off"}'])
    plan, _ = plan_image(vlm, TraceCollector(), small_image(), {}, policy=policy(mem))
    assert plan.restorer == "none" and plan.use_restored is False


def test_policy_decides_only_the_nodes_with_strong_evidence() -> None:
    mem = memory(("sr", "off", 0.9, 50), ("sr", "auto", 0.1, 50))
    vlm = FakeVLM([profile().model_dump_json(), '{"restorer": "dehaze"}', '{"choice": "original"}'])
    plan, _ = plan_image(vlm, TraceCollector(), small_image(), restorers(), policy=policy(mem))
    assert len(vlm.prompts) == 3 and plan.sr_factor is None


# ---- detector pair with a preferred set -------------------------------------------------------

CATALOG = [
    DetectorInfo(name=n, capabilities="x", speed_class="fast") for n in ("a", "b", "c")
]
K = SETTINGS.thresholds.agent.top_k_detectors


def test_detector_select_uses_preferred_without_a_vlm_call() -> None:
    vlm, trace = FakeVLM([]), TraceCollector()
    out = detector_select(
        vlm, trace, ["car"], profile(), CATALOG, ["c", "b", "a"], "some experience", SETTINGS,
        preferred=["a", "b"],
    )
    assert out.detectors == ["a", "b"] and vlm.prompts == []
    assert "experience policy" in trace.events[0].rationale


@pytest.mark.parametrize("preferred", [["a"], ["a", "zzz"], ["a", "a"], ["a", "b", "c"]])
def test_detector_select_ignores_a_preferred_set_that_is_not_k_catalog_detectors(
    preferred: list[str],
) -> None:
    out = detector_select(
        FakeVLM([]), TraceCollector(), ["car"], profile(), CATALOG, ["c", "b", "a"], "",
        SETTINGS, preferred=preferred,
    )
    assert out.detectors == ["c", "b"][:K]


def test_detect_image_runs_only_the_preferred_detectors_with_no_vlm_call() -> None:
    def car(detector: str, x: float) -> Detection:
        return Detection(box=Box(x, 10, x + 30, 40), label="car", score=0.9, detector=detector)

    detectors = {
        "a": FakeDetector("a", [car("a", 10)]),
        "b": FakeDetector("b", [car("b", 100)]),
        "c": FakeDetector("c", [car("c", 200)]),
    }
    vlm = FakeVLM([])
    found = detect_image(
        vlm, TraceCollector(), np.zeros((60, 300, 3), np.uint8), ["car"], profile(), detectors,
        CATALOG, adjudicate=False, settings=SETTINGS, priority=["c", "b", "a"],
        experience="Similar scenes (1): a+b F1 0.80 (n=50)", preferred=["a", "b"],
    )
    assert vlm.prompts == [] and sorted(d.detector for d in found) == ["a", "b"]
