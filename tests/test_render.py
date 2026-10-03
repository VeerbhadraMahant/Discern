import json
from pathlib import Path

import av
import numpy as np
import pytest

from discern.video.io import iter_sampled_frames, probe
from discern.video.render import render_video, tracks_to_json
from discern.video.types import Track
from discern.vision.boxes import Box
from tests.synth_video import background, moving_box_frames, write_clip


def sample_track(status: str = "accepted") -> Track:
    # box on sampled frames 0 and 10 (0.0 s and 1.0 s); interpolated in between
    return Track(
        id=4,
        frames={0: Box(20, 20, 60, 60), 10: Box(60, 20, 100, 60)},
        timestamps={0: 0.0, 10: 1.0},
        scores={0: 0.9, 10: 0.8},
        label="car",
        status=status,  # type: ignore[arg-type]
        rationale="clear",
    )


def decode(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]


def test_render_produces_a_decodable_video_with_the_same_frame_count(tmp_path: Path) -> None:
    src = write_clip(tmp_path / "src.mp4", [background()] * 20, fps=10)
    dst = tmp_path / "out.mp4"
    assert render_video(src, [sample_track()], dst) == 20
    frames = decode(dst)
    assert len(frames) == 20
    info = probe(dst)
    assert (info.width, info.height, info.fps) == (160, 96, pytest.approx(10.0))


def test_render_draws_the_interpolated_box_only_while_the_track_exists(tmp_path: Path) -> None:
    src = write_clip(tmp_path / "src.mp4", [background(level=60)] * 20, fps=10)
    dst = tmp_path / "out.mp4"
    render_video(src, [sample_track()], dst)
    frames = decode(dst)

    def coloured(frame: np.ndarray) -> int:  # pixels near track 4's magenta
        return int((np.abs(frame.astype(int) - (220, 80, 220)).sum(axis=2) < 90).sum())

    assert coloured(frames[0]) > 50  # outline at t = 0
    assert coloured(frames[5]) > 50  # interpolated at t = 0.5
    assert coloured(frames[15]) == 0  # track ended at t = 1.0


def test_rejected_tracks_are_not_drawn(tmp_path: Path) -> None:
    src = write_clip(tmp_path / "src.mp4", [background(level=60)] * 10, fps=10)
    plain, drawn = tmp_path / "plain.mp4", tmp_path / "drawn.mp4"
    render_video(src, [sample_track("rejected")], plain)
    render_video(src, [sample_track()], drawn)
    a, b = decode(plain)[0], decode(drawn)[0]
    box_region = (slice(18, 24), slice(20, 60))
    assert np.abs(a[box_region].astype(int) - b[box_region].astype(int)).mean() > 5
    ref = decode(src)[0]
    assert np.abs(a[60:90].astype(int) - ref[60:90].astype(int)).mean() < 6  # no box drawn


def test_render_without_tracks_keeps_the_frame_count(tmp_path: Path) -> None:
    src = write_clip(tmp_path / "src.mp4", [background()] * 5, fps=10)
    assert render_video(src, [], tmp_path / "out.mp4") == 5


def test_tracks_to_json_round_trips_without_crops() -> None:
    data = json.loads(tracks_to_json([sample_track()]))
    assert data[0]["id"] == 4
    assert data[0]["label"] == "car"
    assert data[0]["status"] == "accepted"
    assert data[0]["frames"]["10"] == [60.0, 20.0, 100.0, 60.0]
    assert data[0]["timestamps"]["10"] == 1.0
    assert (data[0]["t_start"], data[0]["t_end"]) == (0.0, 1.0)
    assert "best_crops" not in data[0]


def test_sampled_frames_of_rendered_video_still_decode(tmp_path: Path) -> None:
    src = write_clip(tmp_path / "src.mp4", moving_box_frames(20), fps=10)
    dst = tmp_path / "out.mp4"
    render_video(src, [sample_track()], dst)
    assert len(list(iter_sampled_frames(dst, 5.0, 1280))) == 10


def test_render_handles_odd_dimensions_and_variable_frame_rate(tmp_path: Path) -> None:
    # 161x95 cannot be yuv420p: the clip is yuv444p, the output is cut to 160x94
    times = [100 * i for i in range(10)] + [2500 + 100 * i for i in range(10)]
    frames = [background(0, 60, (161, 95))] * 20
    src = write_clip(tmp_path / "odd_vfr.mp4", frames, fps=10, times_ms=times, pix_fmt="yuv444p")
    dst = tmp_path / "out.mp4"
    track = sample_track().model_copy(
        update={"timestamps": {0: 0.0, 10: 2.5}}  # frame 10 is the first after the stall
    )
    assert render_video(src, [track], dst) == 20
    out = decode(dst)
    assert len(out) == 20
    assert out[0].shape == (94, 160, 3)
    # frame 9 is at 0.9 s, still inside the first track segment (0.0 s to 2.5 s)
    assert int((np.abs(out[9].astype(int) - (220, 80, 220)).sum(axis=2) < 90).sum()) > 50
