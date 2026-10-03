from pathlib import Path

import av
import pytest

from discern.agent.schemas import SceneProfile
from discern.models.fakes import FakeRestorer, FakeVLM
from discern.models.roles import Detection, Image
from discern.trace import TraceCollector
from discern.video.pipeline import ingest_video, track_video
from discern.video.render import render_video
from discern.video.types import Shot, VideoError
from tests.synth_video import (
    BLUE,
    ColourEmbedder,
    background,
    find_box,
    frame_with_box,
    settings_for,
    write_clip,
)

DARK_RED = (110, 10, 10)
BRIGHTEN = 60
LEVEL_DARK, LEVEL_BRIGHT = 15, 200


def profile_json(label: str) -> str:
    return SceneProfile.model_validate(
        {
            "scene_label": label,
            "illumination": "dark" if label == "low_light" else "normal",
            "visibility": "poor",
            "object_scale": "medium",
            "object_density": "sparse",
            "confidence": 0.9,
        }
    ).model_dump_json()


LOWLIGHT = '{"restorer": "lowlight", "rationale": "dark"}'
RESTORED = '{"choice": "restored", "rationale": "clearer"}'
ORIGINAL = '{"choice": "original", "rationale": "no gain"}'
SR_OFF = '{"factor": "off", "rationale": "ok"}'
ACCEPT = '{"accept": true, "label": "car", "rationale": "a car"}'
REJECT = '{"accept": false, "label": null, "rationale": "background"}'


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shot 0: dark scene with a dark red box. Shot 1: bright scene with a blue box."""
    dark = background(1, LEVEL_DARK)
    bright = background(2, LEVEL_BRIGHT)
    frames = [frame_with_box((10 + 4 * i, 30, 34 + 4 * i, 62), DARK_RED, dark) for i in range(30)]
    frames += [frame_with_box((10 + 4 * i, 30, 34 + 4 * i, 62), BLUE, bright) for i in range(30)]
    return write_clip(tmp_path_factory.mktemp("pipe") / "dark_then_bright.mp4", frames, fps=10)


class Detect:
    """Finds the coloured box; records the mean brightness it saw per shot."""

    def __init__(self) -> None:
        self.brightness: dict[int, list[float]] = {}

    def __call__(self, shot: Shot, image: Image) -> list[Detection]:
        self.brightness.setdefault(shot.id, []).append(float(image.mean()))
        box = find_box(image, 0) or find_box(image, 2)
        return [Detection(box=box, label="car", score=0.9, detector="fake")] if box else []


def vlm_script(
    image_choice: str = RESTORED, verdicts: tuple[str, ...] = (ACCEPT, ACCEPT)
) -> FakeVLM:
    plan_dark = [profile_json("low_light"), LOWLIGHT, image_choice, SR_OFF]
    plan_bright = [profile_json("normal"), SR_OFF]
    return FakeVLM([*plan_dark, *plan_bright, *verdicts])


def restorers() -> dict[str, FakeRestorer]:
    return {"lowlight": FakeRestorer("lowlight", offset=BRIGHTEN)}


def test_ingest_plans_once_per_shot_and_caches_the_plan(clip: Path) -> None:
    vlm, trace = vlm_script(), TraceCollector()
    ingest = ingest_video(clip, vlm, trace, restorers(), settings=settings_for())
    assert [s.id for s in ingest.shots] == [0, 1]
    assert ingest.shots[1].t_start == pytest.approx(3.0, abs=0.2)
    assert ingest.plans[0].restorer == "lowlight" and ingest.plans[0].use_restored
    assert ingest.plans[1].restorer == "none" and not ingest.plans[1].use_restored
    assert len(vlm.prompts) == 6  # 4 + 2 plan calls for the whole clip, not per frame
    assert any(e.node == "video_ingest" for e in trace.events)


def test_shot_plan_is_applied_to_every_sampled_frame_of_its_shot(clip: Path) -> None:
    vlm = vlm_script()
    ingest = ingest_video(clip, vlm, TraceCollector(), restorers(), settings=settings_for())
    seen = Detect()
    for shot, frame in ingest.frames():
        seen(shot, frame.image)
    dark, bright = seen.brightness[0], seen.brightness[1]
    assert len(dark) + len(bright) == 30
    assert min(dark) > LEVEL_DARK + BRIGHTEN - 5  # every dark frame was restored
    assert max(bright) < LEVEL_BRIGHT + 5  # untouched (a restored frame would clip near 255)


def test_original_choice_leaves_frames_untouched(clip: Path) -> None:
    ingest = ingest_video(
        clip,
        vlm_script(image_choice=ORIGINAL),
        TraceCollector(),
        restorers(),
        settings=settings_for(),
    )
    assert not ingest.plans[0].use_restored
    seen = Detect()
    for shot, frame in ingest.frames():
        seen(shot, frame.image)
    assert max(seen.brightness[0]) < LEVEL_DARK + 30


def test_ingest_rejects_videos_over_the_profile_limit(clip: Path) -> None:
    settings = settings_for()
    short = settings.model_copy(
        update={"profile": settings.profile.model_copy(update={"max_video_seconds": 2})}
    )
    with pytest.raises(VideoError, match="allows at most"):
        ingest_video(clip, FakeVLM([]), TraceCollector(), {}, settings=short)


def test_end_to_end_ingest_and_tracking_with_fakes(clip: Path, tmp_path: Path) -> None:
    trace = TraceCollector()
    vlm = vlm_script()
    ingest = ingest_video(clip, vlm, trace, restorers(), settings=settings_for(max_long_side=80))
    tracks = track_video(ingest, Detect(), ColourEmbedder(), vlm, trace, ["car"])

    assert len(tracks) == 2  # one per shot; the red and blue objects are not merged
    assert all(t.status == "accepted" and t.label == "car" for t in tracks)
    assert all(t.rationale == "a car" for t in tracks)
    first, second = tracks
    assert first.t_end < second.t_start
    # boxes are in original (160x96) coordinates although detection ran at 80x48
    for track in tracks:
        box = track.frames[min(track.frames)]
        assert box.y1 == pytest.approx(30, abs=3)
        assert box.y2 == pytest.approx(62, abs=3)
        assert box.x2 - box.x1 == pytest.approx(24, abs=4)
    nodes = [e.node for e in trace.events]
    assert nodes.count("adjudicate_track") == 2  # one VLM call per track
    assert "track" in nodes and "reid" in nodes

    assert render_video(clip, tracks, tmp_path / "annotated.mp4") == 60


def test_rejected_tracks_stay_in_the_result_with_their_rationale(clip: Path) -> None:
    trace = TraceCollector()
    vlm = vlm_script(verdicts=(ACCEPT, REJECT))
    ingest = ingest_video(clip, vlm, trace, restorers(), settings=settings_for())
    tracks = track_video(ingest, Detect(), ColourEmbedder(), vlm, trace, ["car"])
    assert [t.status for t in tracks] == ["accepted", "rejected"]
    assert tracks[1].rationale == "background"


def test_bad_vlm_output_uses_the_deterministic_fallback(clip: Path) -> None:
    trace = TraceCollector()
    vlm = vlm_script(verdicts=("x", "y", "x", "y"))
    ingest = ingest_video(clip, vlm, trace, restorers(), settings=settings_for())
    tracks = track_video(ingest, Detect(), ColourEmbedder(), vlm, trace, ["car"])
    assert all(t.status == "accepted" for t in tracks)  # detector score 0.9 >= fallback threshold
    assert sum(e.fallback_used for e in trace.events if e.node == "adjudicate_track") == 2


def test_no_detections_gives_no_tracks(clip: Path) -> None:
    trace = TraceCollector()
    vlm = vlm_script(verdicts=())
    ingest = ingest_video(clip, vlm, trace, restorers(), settings=settings_for())
    tracks = track_video(ingest, lambda s, i: [], ColourEmbedder(), vlm, trace, ["car"])
    assert tracks == []


def test_variable_frame_rate_clip_with_a_start_offset_round_trips_ingest_track_and_render(
    tmp_path: Path,
) -> None:
    # container timeline starts at 2.0 s and stalls for 1.5 s after frame 19
    times = [2000 + 100 * i for i in range(20)] + [5500 + 100 * i for i in range(20)]
    bg = background(3, 90)
    frames = [frame_with_box((10 + 3 * i, 30, 34 + 3 * i, 62), bg=bg) for i in range(40)]
    clip = write_clip(tmp_path / "vfr.mp4", frames, fps=10, times_ms=times)
    trace = TraceCollector()
    vlm = FakeVLM([profile_json("normal"), SR_OFF, ACCEPT])
    ingest = ingest_video(clip, vlm, trace, {}, settings=settings_for())
    (track,) = track_video(ingest, Detect(), ColourEmbedder(), vlm, trace, ["car"])

    assert sorted(track.frames) == list(range(0, 40, 2))  # even frames: the 5 fps stride
    for index, box in track.frames.items():  # the box found in frame `index` is that frame's
        assert box.x1 == pytest.approx(10 + 3 * index, abs=3)
        assert track.timestamps[index] == pytest.approx(times[index] / 1000, abs=1e-3)

    out = tmp_path / "annotated.mp4"
    assert render_video(clip, [track], out) == 40
    with av.open(str(out)) as container:
        rendered = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    drawn = find_box(rendered[30], channel=1)  # track 1 is drawn in green; frame 30 is sampled
    assert drawn is not None and drawn.x1 == pytest.approx(10 + 3 * 30, abs=4)
