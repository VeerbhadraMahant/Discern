from pathlib import Path

import pytest

from discern.video.io import (
    decode_limit,
    iter_sampled_frames,
    probe,
    validate,
    working_size,
)
from discern.video.types import VideoError
from discern.vision.boxes import Box
from tests.synth_video import (
    BLUE,
    SIZE,
    background,
    find_box,
    frame_with_box,
    moving_box_frames,
    settings_for,
    write_clip,
)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    frames = moving_box_frames(30) + moving_box_frames(30, colour=BLUE, bg=background(5, 200))
    return write_clip(tmp_path_factory.mktemp("io") / "two_shots.mp4", frames, fps=10)


def test_probe_reports_fps_duration_size_and_frames(clip: Path) -> None:
    info = probe(clip)
    assert info.fps == pytest.approx(10.0)
    assert info.duration == pytest.approx(6.0, abs=0.15)
    assert (info.width, info.height) == SIZE
    assert info.frame_count == 60


def test_sampling_follows_sample_fps(clip: Path) -> None:
    frames = list(iter_sampled_frames(clip, 5.0, 1280))
    assert len(frames) == 30
    assert [f.index for f in frames[:4]] == [0, 2, 4, 6]
    assert frames[3].time == pytest.approx(0.6, abs=1e-3)
    assert frames[0].image.shape == (SIZE[1], SIZE[0], 3)
    assert frames[0].scale_factor == 1.0


def test_sample_fps_above_native_keeps_every_frame(clip: Path) -> None:
    assert len(list(iter_sampled_frames(clip, 50.0, 1280))) == 60


def test_downscale_keeps_long_side_and_maps_boxes_back(clip: Path) -> None:
    frame = next(iter_sampled_frames(clip, 5.0, 80))
    assert frame.image.shape == (48, 80, 3)
    assert frame.original_size == SIZE
    assert frame.scale_factor == 0.5
    assert frame.to_original(Box(10, 10, 20, 20)) == Box(20, 20, 40, 40)
    found = find_box(frame.image)
    assert found is not None
    mapped = frame.to_original(found)
    # the red rectangle is (10, 30, 34, 62) in the original frame
    for got, want in zip(mapped, (10, 30, 34, 62), strict=True):
        assert got == pytest.approx(want, abs=3)


def test_working_size_never_upscales() -> None:
    assert working_size(1920, 1080, 1280) == (1280, 720)
    assert working_size(160, 96, 1280) == (160, 96)


def test_validate_rejects_long_videos(clip: Path) -> None:
    settings = settings_for()
    validate(probe(clip), settings.profile)
    short = settings.profile.model_copy(update={"max_video_seconds": 3})
    with pytest.raises(VideoError, match="allows at most 3s"):
        validate(probe(clip), short)


def test_decoding_stops_when_decoded_time_exceeds_the_limit(clip: Path) -> None:
    with pytest.raises(VideoError, match="runs past 3s"):
        list(iter_sampled_frames(clip, 5.0, 1280, max_seconds=3.0))
    assert len(list(iter_sampled_frames(clip, 5.0, 1280, max_seconds=10.0))) == 30


def test_decode_limit_is_max_duration_plus_the_configured_margin() -> None:
    settings = settings_for(duration_margin_seconds=2.5)
    assert decode_limit(settings) == settings.profile.max_video_seconds + 2.5


def test_missing_and_corrupt_files_raise_clear_errors(tmp_path: Path) -> None:
    with pytest.raises(VideoError, match="not found"):
        probe(tmp_path / "nope.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video")
    with pytest.raises(VideoError, match="cannot open"):
        probe(junk)


def test_unusual_file_names_are_passed_to_the_decoder_verbatim(
    clip: Path, tmp_path: Path
) -> None:
    odd = tmp_path / "a b; echo $(whoami) & x.mp4"
    odd.write_bytes(clip.read_bytes())
    assert probe(odd).frame_count == 60


def test_sampling_stride_holds_after_a_timestamp_gap_and_with_a_start_offset(
    tmp_path: Path,
) -> None:
    # variable frame rate: 10 fps, a 1.5 s stall before frame 20, then 10 fps again; the whole
    # clip starts at 2.0 s on the container timeline
    times = [2000 + 100 * i for i in range(20)] + [5500 + 100 * i for i in range(20)]
    clip = write_clip(tmp_path / "vfr.mp4", moving_box_frames(40), fps=10, times_ms=times)
    frames = list(iter_sampled_frames(clip, 5.0, 1280))
    gaps = [b.time - a.time for a, b in zip(frames, frames[1:], strict=False)]
    assert frames[0].index == 0
    assert all(g >= 0.2 - 1e-3 for g in gaps), [round(f.time, 2) for f in frames]
    assert len(frames) == 10 + 10  # 2 s of video at 5 fps before the stall, 2 s after


def test_odd_dimensions_decode_downscale_and_map_back(tmp_path: Path) -> None:
    size = (161, 95)
    frames = [frame_with_box((20, 20, 60, 70), bg=background(0, 90, size)) for _ in range(10)]
    clip = write_clip(tmp_path / "odd.mp4", frames, fps=10, pix_fmt="yuv444p")
    info = probe(clip)
    assert (info.width, info.height) == size
    frame = next(iter_sampled_frames(clip, 5.0, 80))
    assert frame.image.shape == (47, 80, 3)  # long side 80, never zero-sized
    assert frame.original_size == size
    found = find_box(frame.image)
    assert found is not None
    for got, want in zip(frame.to_original(found), (20, 20, 60, 70), strict=True):
        assert got == pytest.approx(want, abs=3)
