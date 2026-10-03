from pathlib import Path

import numpy as np
import pytest

from discern.video.io import iter_sampled_frames, probe
from discern.video.shots import FrameScan, build_shots, detect_shots
from discern.video.types import Shot
from discern.vision.stats import laplacian_variance
from tests.synth_video import BLUE, background, moving_box_frames, settings_for, write_clip


@pytest.fixture(scope="module")
def two_shot_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    frames = moving_box_frames(30) + moving_box_frames(30, colour=BLUE, bg=background(5, 200))
    return write_clip(tmp_path_factory.mktemp("shots") / "cut.mp4", frames, fps=10)


def shots_of(path: Path, **video: float) -> list[Shot]:
    settings = settings_for(**video)
    frames = iter_sampled_frames(path, 5.0, 1280)
    return detect_shots(frames, 5.0, probe(path).duration, settings.thresholds.video)


def test_hard_cut_gives_two_shots(two_shot_clip: Path) -> None:
    shots = shots_of(two_shot_clip)
    assert [s.id for s in shots] == [0, 1]
    assert shots[0].t_start == 0.0
    assert shots[1].t_start == pytest.approx(3.0, abs=0.2)
    assert shots[0].t_end == shots[1].t_start
    for s in shots:
        assert s.t_start <= s.keyframe_time < s.t_end


def test_clip_without_cuts_is_one_shot(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "still.mp4", moving_box_frames(40), fps=10)
    shots = shots_of(clip)
    assert len(shots) == 1
    assert shots[0].t_start == 0.0


def test_min_shot_seconds_suppresses_a_cut_right_after_the_start(tmp_path: Path) -> None:
    frames = moving_box_frames(4) + moving_box_frames(36, colour=BLUE, bg=background(5, 200))
    clip = write_clip(tmp_path / "early.mp4", frames, fps=10)
    assert len(shots_of(clip, min_shot_seconds=0.2)) == 2
    assert len(shots_of(clip, min_shot_seconds=1.0)) == 1


def test_single_image_is_one_shot() -> None:
    shots = build_shots([FrameScan(index=0, time=0.0, sharpness=1.0)], [], end_time=0.0)
    assert len(shots) == 1
    assert shots[0].keyframe_index == 0


def test_no_frames_gives_no_shots() -> None:
    assert build_shots([], [], end_time=1.0) == []


def scans(sharpness: list[float]) -> list[FrameScan]:
    return [FrameScan(index=i, time=i * 0.5, sharpness=s) for i, s in enumerate(sharpness)]


def test_keyframe_is_the_sharpest_frame_near_the_midpoint() -> None:
    # 9 frames over 4.5 s: midpoint 2.25 s, window +-1.125 s covers frames 3..6 (1.5..3.0 s)
    sharp = scans([1, 1, 1, 2, 3, 5, 1, 1, 1])
    (shot,) = build_shots(sharp, [], end_time=4.5)
    assert shot.keyframe_index == 5


def test_keyframe_ignores_a_sharper_frame_far_from_the_midpoint() -> None:
    far = scans([100, 1, 1, 2, 5, 3, 1, 1, 1])
    (shot,) = build_shots(far, [], end_time=4.5)
    assert shot.keyframe_index == 4


def test_cuts_split_scans_and_each_shot_gets_a_keyframe() -> None:
    (a, b) = build_shots(scans([1, 3, 2, 4, 1, 9]), [3], end_time=3.0)
    assert (a.t_start, a.t_end, b.t_start, b.t_end) == (0.0, 1.5, 1.5, 3.0)
    assert a.keyframe_index == 1
    assert b.keyframe_index == 5


def test_laplacian_variance_ranks_sharp_above_blurred() -> None:
    sharp = background(1)
    blurred = np.full_like(sharp, 90)
    assert laplacian_variance(sharp) > laplacian_variance(blurred) == 0.0


@pytest.mark.parametrize(("cut_frame", "first_new_sample"), [(29, 3.0), (30, 3.0), (31, 3.2)])
def test_a_shot_starts_at_the_first_sampled_frame_after_the_cut(
    tmp_path: Path, cut_frame: int, first_new_sample: float
) -> None:
    frames = moving_box_frames(cut_frame) + moving_box_frames(
        60 - cut_frame, colour=BLUE, bg=background(5, 200)
    )
    clip = write_clip(tmp_path / "cut.mp4", frames, fps=10)
    shots = shots_of(clip)
    assert [s.t_start for s in shots] == pytest.approx([0.0, first_new_sample])
