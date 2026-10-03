from collections.abc import Sequence
from pathlib import Path

import numpy as np

from discern.agent.llm_io import load_prompt
from discern.agent.nodes.adjudicate_track import adjudicate_track
from discern.agent.schemas import TrackAdjudication
from discern.config.settings import load_settings
from discern.models.fakes import FakeVLM
from discern.models.roles import Image
from discern.trace import TraceCollector
from discern.video.types import Track, TrackCrop
from discern.vision.boxes import Box

SNAPSHOTS = Path(__file__).parent / "snapshots"
SETTINGS = load_settings()
FALLBACK = SETTINGS.thresholds.agent.fallback_accept_score
TARGETS = ["car", "person"]


def make_track(score: float = 0.9, crops: int = 3) -> Track:
    return Track(
        id=1,
        frames={0: Box(0, 0, 10, 10), 2: Box(2, 0, 12, 10)},
        timestamps={0: 0.0, 2: 0.4},
        scores={0: score, 2: score},
        label="car",
        best_crops=[
            TrackCrop(
                frame_index=i,
                time=0.2 * i,
                box=Box(0, 0, 10, 10),
                rank=float(100 - i),
                image=np.full((12, 14, 3), 50 * (i + 1), dtype=np.uint8),
            )
            for i in range(crops)
        ],
    )


def accept(label: str = "car") -> str:
    return f'{{"accept": true, "label": "{label}", "rationale": "clear car"}}'


REJECT = '{"accept": false, "label": null, "rationale": "background"}'


def test_prompt_matches_golden_snapshot() -> None:
    rendered = load_prompt("adjudicate_track", 1).render(
        targets="car, person",
        track="2 sampled frames from 0.00s to 0.40s, detector label car, mean score 0.90",
        crops="1. t=0.00s, box area x score x sharpness rank=100\n2. t=0.20s, rank=99",
    )
    assert rendered == (SNAPSHOTS / "adjudicate_track.v1.txt").read_text(encoding="utf-8")


def test_accept_with_label() -> None:
    result = adjudicate_track(FakeVLM([accept("person")]), TraceCollector(), make_track(), TARGETS)
    assert result.accept and result.label == "person"


def test_reject() -> None:
    result = adjudicate_track(FakeVLM([REJECT]), TraceCollector(), make_track(), TARGETS)
    assert not result.accept and result.label is None


class RecordingVLM(FakeVLM):
    def __init__(self, responses: Sequence[str]) -> None:
        super().__init__(responses)
        self.images: list[Sequence[Image]] = []

    def generate(self, prompt: str, images: Sequence[Image] = ()) -> str:
        self.images.append(images)
        return super().generate(prompt, images)


def test_one_call_with_at_most_three_numbered_crops() -> None:
    vlm = RecordingVLM([accept()])
    trace = TraceCollector()
    track = make_track(crops=5)
    adjudicate_track(vlm, trace, track, TARGETS)
    assert len(vlm.images) == 1 and len(vlm.images[0]) == 3
    assert "3. t=" in vlm.prompts[0] and "4. t=" not in vlm.prompts[0]
    # the number is drawn onto the crop, so it differs from the stored one
    assert not np.array_equal(vlm.images[0][0], track.best_crops[0].image)
    assert trace.events[0].node == "adjudicate_track"
    assert trace.events[0].prompt_version == "adjudicate_track@v1"


def test_malformed_output_falls_back_to_accept_when_score_is_high() -> None:
    trace = TraceCollector()
    result = adjudicate_track(FakeVLM(["x", "y"]), trace, make_track(score=FALLBACK), TARGETS)
    assert result == TrackAdjudication(
        accept=True, label="car", rationale="fallback: mean detector score threshold"
    )
    assert trace.events[0].fallback_used is True


def test_malformed_output_falls_back_to_reject_when_score_is_low() -> None:
    result = adjudicate_track(FakeVLM(["x", "y"]), TraceCollector(), make_track(score=0.1), TARGETS)
    assert not result.accept and result.label is None


def test_label_outside_targets_is_caught_by_code() -> None:
    trace = TraceCollector()
    result = adjudicate_track(FakeVLM([accept("bicycle")]), trace, make_track(), TARGETS)
    assert result.accept and result.label == "car"  # high-score fallback with the majority label
    assert trace.events[-1].node == "adjudicate_track.guard"
    assert trace.events[-1].fallback_used is True


def test_accept_without_label_is_caught_by_code() -> None:
    bad = '{"accept": true, "label": null, "rationale": "r"}'
    result = adjudicate_track(FakeVLM([bad]), TraceCollector(), make_track(score=0.1), TARGETS)
    assert not result.accept
