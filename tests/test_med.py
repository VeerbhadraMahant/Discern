from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from discern.agent.llm_io import load_prompt
from discern.agent.med import detect_image
from discern.agent.nodes.adjudicate import AdjudicatedDetection, adjudicate
from discern.agent.nodes.detector_select import detector_select
from discern.agent.schemas import DetectorInfo, SceneProfile
from discern.config.settings import load_settings
from discern.models.fakes import FakeDetector, FakeVLM
from discern.models.roles import Detection, Image
from discern.trace import TraceCollector
from discern.vision.boxes import Box
from discern.vision.grouping import InstanceGroup

SNAPSHOTS = Path(__file__).parent / "snapshots"
SETTINGS = load_settings()
K = SETTINGS.thresholds.agent.top_k_detectors
TARGETS = ["car", "person"]

CATALOG = [
    DetectorInfo(name="fast_det", capabilities="quick, good on large objects", speed_class="fast"),
    DetectorInfo(name="acc_det", capabilities="accurate, small objects", speed_class="fast"),
    DetectorInfo(name="dense_det", capabilities="dense scenes", speed_class="slow"),
]
PRIORITY = ["acc_det", "fast_det", "dense_det"]

PROFILE = SceneProfile.model_validate(
    {
        "scene_label": "normal",
        "illumination": "normal",
        "visibility": "clear",
        "object_scale": "small",
        "object_density": "dense",
        "confidence": 0.8,
    }
)


def det(
    box: tuple[float, float, float, float], score: float, detector: str, label: str = "car"
) -> Detection:
    return Detection(box=Box(*box), label=label, score=score, detector=detector)


def image() -> Image:
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img[30:70, 40:80] = (230, 40, 40)
    img[30:70, 200:240] = (40, 40, 230)
    return img


def choice_json(*names: str) -> str:
    return '{"detectors": [' + ", ".join(f'"{n}"' for n in names) + '], "rationale": "r"}'


def pick(candidate: int, label: str = "car") -> str:
    return f'{{"candidate": {candidate}, "label": "{label}", "reject": false, "rationale": "r"}}'


REJECT = '{"candidate": null, "label": null, "reject": true, "rationale": "background"}'

# ---- prompt snapshots -------------------------------------------------------------------------

PROMPT_VARIABLES: dict[str, dict[str, object]] = {
    "detector_select": {
        "targets": "car, person",
        "profile": '{"scene_label": "normal"}',
        "catalog": "- fast_det (fast): quick\n- acc_det (fast): accurate",
        "k": 2,
        "experience": "",
    },
    "adjudicate": {
        "targets": "car, person",
        "candidates": "1. detector=a, label=car, score=0.90\n2. detector=b, label=car, score=0.60",
    },
}


@pytest.mark.parametrize("name", sorted(PROMPT_VARIABLES))
def test_prompt_matches_golden_snapshot(name: str) -> None:
    rendered = load_prompt(name, 1).render(**PROMPT_VARIABLES[name])
    assert rendered == (SNAPSHOTS / f"{name}.v1.txt").read_text(encoding="utf-8")


# ---- detector_select --------------------------------------------------------------------------


def select(vlm: FakeVLM, trace: TraceCollector | None = None) -> list[str]:
    return detector_select(
        vlm, trace or TraceCollector(), TARGETS, PROFILE, CATALOG, PRIORITY, settings=SETTINGS
    ).detectors


def test_select_valid_choice_is_kept() -> None:
    assert select(FakeVLM([choice_json("dense_det", "fast_det")])) == ["dense_det", "fast_det"]


def test_select_drops_unknown_and_duplicates_and_pads() -> None:
    trace = TraceCollector()
    names = select(FakeVLM([choice_json("ghost", "dense_det", "dense_det")]), trace)
    assert names == ["dense_det", "acc_det"][:K]
    assert any(e.node == "detector_select.guard" for e in trace.events)


def test_select_truncates_to_top_k() -> None:
    assert len(select(FakeVLM([choice_json("fast_det", "acc_det", "dense_det")]))) == K


def test_select_invalid_json_falls_back_to_priority() -> None:
    trace = TraceCollector()
    assert select(FakeVLM(["nope", "still nope"]), trace) == PRIORITY[:K]
    assert trace.events[0].fallback_used


# ---- adjudicate -------------------------------------------------------------------------------


def group_of(*dets: Detection) -> InstanceGroup:
    return InstanceGroup(dets[0], tuple(dets))


A = det((40, 30, 80, 70), 0.9, "acc_det")
B = det((42, 32, 82, 72), 0.6, "fast_det")


def run_adjudicate(
    vlm: FakeVLM, group: InstanceGroup, trace: TraceCollector | None = None
) -> AdjudicatedDetection | None:
    return adjudicate(vlm, trace or TraceCollector(), image(), group, TARGETS, SETTINGS)


def test_adjudicate_picks_second_candidate_and_relabels() -> None:
    vlm = FakeVLM([pick(2, "person")])
    result = run_adjudicate(vlm, group_of(A, B))
    assert result is not None
    assert (result.box, result.label, result.source_detector) == (B.box, "person", "fast_det")
    assert "detector=fast_det, label=car, score=0.60" in vlm.prompts[0]


def test_adjudicate_reject_returns_none() -> None:
    assert run_adjudicate(FakeVLM([REJECT]), group_of(A, B)) is None


@pytest.mark.parametrize(
    "answer",
    [
        pick(3),  # out of range
        pick(1, "dog"),  # label not in targets
        '{"candidate": 1, "label": "car", "reject": true}',  # reject with candidate
        '{"candidate": null, "label": null, "reject": false}',  # neither
    ],
)
def test_adjudicate_invalid_answer_uses_fallback(answer: str) -> None:
    trace = TraceCollector()
    result = run_adjudicate(FakeVLM([answer]), group_of(A, B), trace)
    assert result is not None and result.box == A.box
    assert any(e.node == "adjudicate.guard" for e in trace.events)


def test_adjudicate_invalid_json_fallback_rejects_low_score_anchor() -> None:
    weak = det((40, 30, 80, 70), SETTINGS.thresholds.agent.fallback_accept_score - 0.01, "x")
    assert run_adjudicate(FakeVLM(["bad", "bad"]), group_of(weak)) is None


def test_adjudicate_sends_one_annotated_crop() -> None:
    sent: list[Image] = []

    class Spy(FakeVLM):
        def generate(self, prompt: str, images: Sequence[Image] = ()) -> str:
            sent.extend(images)
            return super().generate(prompt, images)

    run_adjudicate(Spy([REJECT]), group_of(A, B))
    assert len(sent) == 1
    assert sent[0].shape[0] < image().shape[0]
    assert (sent[0][..., 0] == 255).any()  # candidate 1 drawn in red


# ---- detect_image -----------------------------------------------------------------------------


def detectors(**kwargs: list[Detection]) -> dict[str, FakeDetector]:
    return {n: FakeDetector(n, d) for n, d in kwargs.items()}


def run(
    vlm: FakeVLM,
    dets: dict[str, FakeDetector],
    adjudicate_on: bool = True,
    thresholds: dict[str, float] | None = None,
) -> list[Detection]:
    return detect_image(
        vlm,
        TraceCollector(),
        image(),
        TARGETS,
        PROFILE,
        dets,  # type: ignore[arg-type]
        CATALOG,
        adjudicate=adjudicate_on,
        settings=SETTINGS,
        operating_thresholds=thresholds,
        priority=PRIORITY,
    )


def test_two_detectors_agreeing_yield_one_detection() -> None:
    dets = detectors(acc_det=[A], fast_det=[B])
    assert run(FakeVLM([choice_json("acc_det", "fast_det"), pick(1)]), dets) == [A]


def test_false_positive_rejected_by_adjudication() -> None:
    fp = det((200, 30, 240, 70), 0.5, "fast_det")
    dets = detectors(acc_det=[A], fast_det=[fp])
    out = run(FakeVLM([choice_json("acc_det", "fast_det"), pick(1), REJECT]), dets)
    assert out == [A]


def test_cheap_fusion_returns_anchors_without_adjudication_calls() -> None:
    dets = detectors(acc_det=[A], fast_det=[B])
    vlm = FakeVLM([choice_json("acc_det", "fast_det")])
    assert run(vlm, dets, adjudicate_on=False) == [A]
    assert len(vlm.prompts) == 1


def test_invalid_json_everywhere_falls_back_deterministically() -> None:
    dets = detectors(acc_det=[A], fast_det=[B])
    out = run(FakeVLM(["x"] * 4), dets)  # select (2 tries) + adjudicate (2 tries)
    assert out == [A]


def test_operating_thresholds_are_per_detector() -> None:
    low = det((200, 30, 240, 70), 0.3, "fast_det")
    dets = detectors(acc_det=[A], fast_det=[low])
    vlm = FakeVLM([choice_json("acc_det", "fast_det")])
    assert run(vlm, dets, adjudicate_on=False, thresholds={"fast_det": 0.5}) == [A]
    vlm = FakeVLM([choice_json("acc_det", "fast_det")])
    assert run(vlm, dets, adjudicate_on=False) == [A, low]  # default 0.25 keeps 0.3


def test_top_k_selection_runs_only_chosen_detectors() -> None:
    other = det((200, 30, 240, 70), 0.9, "dense_det")
    dets = detectors(acc_det=[A], fast_det=[B], dense_det=[other])
    vlm = FakeVLM([choice_json("acc_det", "fast_det", "dense_det")])
    assert run(vlm, dets, adjudicate_on=False) == [A]


def test_missing_detector_is_tolerated() -> None:
    dets = detectors(acc_det=[A])  # fast_det and dense_det not loaded
    assert run(FakeVLM([choice_json("acc_det", "fast_det"), pick(1)]), dets) == [A]
    assert run(FakeVLM(["x", "x"]), {}) == []


def test_adjudicate_all_false_skips_vlm_for_groups_two_detectors_agree_on() -> None:
    dets = detectors(acc_det=[A], fast_det=[B])  # A and B overlap and share a label
    vlm = FakeVLM([choice_json("acc_det", "fast_det")])
    out = detect_image(
        vlm,
        TraceCollector(),
        image(),
        TARGETS,
        PROFILE,
        dets,  # type: ignore[arg-type]
        CATALOG,
        adjudicate_all=False,
        settings=SETTINGS,
        priority=PRIORITY,
    )
    assert out == [A]
    assert len(vlm.prompts) == 1  # detector_select only
