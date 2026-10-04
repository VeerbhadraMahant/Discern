import json
from pathlib import Path

import numpy as np
import pytest

from discern.agent.llm_io import load_prompt
from discern.agent.nodes.image_select import image_select
from discern.agent.nodes.perception import perception
from discern.agent.nodes.restorer_select import restorer_select
from discern.agent.nodes.sr_select import required_factor, sr_select
from discern.agent.sair import plan_image
from discern.agent.schemas import SceneProfile
from discern.config.settings import load_settings
from discern.models.fakes import FakeRestorer, FakeVLM
from discern.models.roles import Image
from discern.trace import TraceCollector

SNAPSHOTS = Path(__file__).parent / "snapshots"
TARGET = load_settings().thresholds.agent.sr_target_long_side


def make_profile(label: str = "fog") -> SceneProfile:
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


def profile_json(label: str = "fog") -> str:
    return make_profile(label).model_dump_json()


def small_image() -> Image:
    return np.full((64, 96, 3), 100, dtype=np.uint8)


def wide_image() -> Image:
    return np.zeros((8, TARGET, 3), dtype=np.uint8)


RESTORED = '{"choice": "restored", "rationale": "edges clearer"}'
ORIGINAL = '{"choice": "original", "rationale": "no gain"}'
SR_ON = '{"factor": 4, "rationale": "small objects"}'
SR_OFF = '{"factor": "off", "rationale": "fine as is"}'


def dehaze_choice() -> str:
    return '{"restorer": "dehaze", "rationale": "haze"}'


# ---- prompt snapshots -------------------------------------------------------------------------

PROMPT_VARIABLES: dict[str, dict[str, object]] = {
    "perception": {},
    "restorer_select": {
        "profile": '{"scene_label": "fog"}',
        "default": "dehaze",
        "experience": "Similar scenes (3): dehaze F1 0.52 (n=14) vs none 0.47 (n=14)",
    },
    "image_select": {"profile": '{"scene_label": "fog"}'},
    "sr_select": {
        "profile": '{"scene_label": "fog"}',
        "width": 640,
        "height": 480,
        "target": 2048,
        "factor": 4,
        "experience": "",
    },
}


@pytest.mark.parametrize("name", sorted(PROMPT_VARIABLES))
def test_prompt_matches_golden_snapshot(name: str) -> None:
    rendered = load_prompt(name, 1).render(**PROMPT_VARIABLES[name])
    assert rendered == (SNAPSHOTS / f"{name}.v1.txt").read_text(encoding="utf-8")


def test_image_select_prompt_uses_paper_criteria() -> None:
    text = load_prompt("image_select", 1).template
    for phrase in ("object visibility", "boundary clarity", "structural integrity", "aesthetics"):
        assert phrase in text
    assert "Image A" in text and "Image B" in text


# ---- perception -------------------------------------------------------------------------------


def test_perception_valid() -> None:
    trace = TraceCollector()
    out = perception(FakeVLM([profile_json("rain")]), trace, small_image())
    assert out.scene_label == "rain"
    assert trace.events[0].node == "perception"


def test_perception_invalid_falls_back_to_stats() -> None:
    trace = TraceCollector()
    out = perception(FakeVLM(["x", "y"]), trace, small_image())
    assert out.scene_label == "normal"  # flat mid-gray image
    assert trace.events[0].fallback_used is True


# ---- restorer_select --------------------------------------------------------------------------


def test_restorer_select_experience_is_injected() -> None:
    vlm = FakeVLM([dehaze_choice()])
    out = restorer_select(vlm, TraceCollector(), make_profile("fog"), experience="EXP-TABLE")
    assert out.restorer == "dehaze"
    assert "EXP-TABLE" in vlm.prompts[0]


def test_restorer_select_vlm_may_pick_none() -> None:
    out = restorer_select(FakeVLM(['{"restorer": "none"}']), TraceCollector(), make_profile())
    assert out.restorer == "none"


def test_restorer_select_invalid_json_uses_rule_map() -> None:
    trace = TraceCollector()
    out = restorer_select(FakeVLM(["x", "y"]), trace, make_profile("low_light"))
    assert out.restorer == "lowlight"
    assert trace.events[0].fallback_used is True


def test_restorer_select_rejects_unmapped_restorer() -> None:
    trace = TraceCollector()
    out = restorer_select(FakeVLM(['{"restorer": "derain"}']), trace, make_profile("fog"))
    assert out.restorer == "dehaze"
    assert trace.events[-1].fallback_used is True


@pytest.mark.parametrize("label", ["normal", "underwater"])
def test_restorer_select_unmapped_scenes_skip_the_vlm(label: str) -> None:
    vlm = FakeVLM([])
    trace = TraceCollector()
    assert restorer_select(vlm, trace, make_profile(label)).restorer == "none"
    assert vlm.prompts == [] and len(trace.events) == 1


EVIDENCE = "Similar scenes (3): lowlight F1 0.39 (n=100) vs none 0.37 (n=100)"


def test_image_select_v2_prompt_matches_golden_snapshot() -> None:
    rendered = load_prompt("image_select", 2).render(
        profile='{"scene_label": "fog"}', experience=EVIDENCE
    )
    assert rendered == (SNAPSHOTS / "image_select.v2.txt").read_text(encoding="utf-8")


def test_image_select_uses_v2_prompt_only_when_experience_is_given() -> None:
    img = small_image()
    plain, informed = FakeVLM([RESTORED]), FakeVLM([RESTORED])
    image_select(plain, TraceCollector(), make_profile(), img, img)
    image_select(informed, TraceCollector(), make_profile(), img, img, EVIDENCE)
    assert "Past experience" not in plain.prompts[0] and EVIDENCE not in plain.prompts[0]
    assert EVIDENCE in informed.prompts[0]


# ---- image_select -----------------------------------------------------------------------------


def test_image_select_restored_and_fallback() -> None:
    img = small_image()
    out = image_select(FakeVLM([RESTORED]), TraceCollector(), make_profile(), img, img)
    assert out.choice == "restored"
    trace = TraceCollector()
    out = image_select(FakeVLM(["x", "y"]), trace, make_profile(), img, img)
    assert out.choice == "original"
    assert trace.events[0].fallback_used is True


# ---- sr_select --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("long_side", "expected"),
    [(TARGET, None), (TARGET + 1, None), (TARGET // 2, 2), (TARGET // 2 - 1, 4), (10, 4)],
)
def test_required_factor_is_clamped_to_2_or_4(long_side: int, expected: int | None) -> None:
    assert required_factor(long_side, TARGET) == expected


def test_sr_factors_have_a_single_definition() -> None:
    from discern.agent.nodes import sr_select as module
    from discern.models import tiling

    assert module.SR_FACTORS is tiling.SR_FACTORS


def test_sr_select_valid_on() -> None:
    out = sr_select(FakeVLM([SR_ON]), TraceCollector(), small_image(), make_profile(), TARGET)
    assert out.factor == 4


def test_sr_select_vlm_can_turn_off() -> None:
    out = sr_select(FakeVLM([SR_OFF]), TraceCollector(), small_image(), make_profile(), TARGET)
    assert out.factor == "off"


def test_sr_select_overrides_wrong_vlm_factor() -> None:
    trace = TraceCollector()
    out = sr_select(
        FakeVLM(['{"factor": 8}']),
        trace,
        np.zeros((TARGET // 2, 8, 3), np.uint8),
        make_profile(),
        TARGET,
    )
    assert out.factor == 2
    assert trace.events[-1].fallback_used is True


def test_sr_select_invalid_json_falls_back_to_on() -> None:
    trace = TraceCollector()
    out = sr_select(FakeVLM(["x", "y"]), trace, small_image(), make_profile(), TARGET)
    assert out.factor == 4
    assert trace.events[0].fallback_used is True


def test_sr_select_skips_vlm_when_target_reached() -> None:
    vlm = FakeVLM([])
    out = sr_select(vlm, TraceCollector(), wide_image(), make_profile(), TARGET)
    assert out.factor == "off" and vlm.prompts == []


# ---- plan_image -------------------------------------------------------------------------------


def restorers() -> dict[str, FakeRestorer]:
    return {"dehaze": FakeRestorer("dehaze", offset=20)}


def test_happy_path_applies_restoration_and_sr() -> None:
    trace = TraceCollector()
    img = small_image()
    vlm = FakeVLM([profile_json("fog"), dehaze_choice(), RESTORED, SR_ON])
    plan, out = plan_image(vlm, trace, img, restorers())
    assert (plan.restorer, plan.use_restored, plan.sr_factor) == ("dehaze", True, 4)
    assert int(out[0, 0, 0]) == 120
    assert [e.node for e in trace.events] == [
        "perception",
        "restorer_select",
        "image_select",
        "sr_select",
    ]
    json.loads(plan.model_dump_json())


def test_restoration_rejected_by_image_select() -> None:
    img = small_image()
    vlm = FakeVLM([profile_json("fog"), dehaze_choice(), ORIGINAL, SR_OFF])
    plan, out = plan_image(vlm, TraceCollector(), img, restorers())
    assert (plan.restorer, plan.use_restored, plan.sr_factor) == ("dehaze", False, None)
    assert out is img


def test_missing_restorer_falls_back_to_none() -> None:
    vlm = FakeVLM([profile_json("rain"), '{"restorer": "derain"}', SR_ON])
    plan, out = plan_image(vlm, TraceCollector(), small_image(), restorers())
    assert plan.restorer == "none" and plan.use_restored is False
    assert plan.sr_factor == 4


def test_failing_restorer_falls_back_to_none() -> None:
    class Broken:
        name = "dehaze"

        def restore(self, image: Image) -> Image:
            raise RuntimeError("out of memory")

    vlm = FakeVLM([profile_json("fog"), dehaze_choice(), SR_ON])
    plan, _ = plan_image(vlm, TraceCollector(), small_image(), {"dehaze": Broken()})
    assert plan.restorer == "none" and plan.use_restored is False


def test_invalid_json_at_every_node_gives_deterministic_plan() -> None:
    trace = TraceCollector()
    dark = np.full((64, 96, 3), 10, dtype=np.uint8)
    vlm = FakeVLM(["bad"] * 8)  # perception, restorer_select, image_select, sr_select: 2 each
    plan, out = plan_image(vlm, trace, dark, {"lowlight": FakeRestorer("lowlight")})
    assert plan.restorer == "lowlight"  # stats -> low_light -> rule map
    assert plan.use_restored is False  # image_select fallback = original
    assert plan.sr_factor == 4  # sr fallback = on
    assert out is dark
    assert all(e.fallback_used for e in trace.events)
    assert len(trace.events) == 4
