import numpy as np

from discern.models.roles import Detection, Image
from discern.video.tracks import MAX_CROPS, BoxInterpolator, TrackBuilder, merge_tracks
from discern.video.types import SampledFrame, Track
from discern.vision.boxes import Box
from tests.synth_video import (
    BLUE,
    RED,
    ColourEmbedder,
    background,
    find_box,
    frame_with_box,
    settings_for,
)

FPS = 5.0
THRESHOLDS = settings_for(
    track_lost_seconds=0.4, reid_cosine=0.8, reid_max_gap_seconds=2.0
).thresholds
VIDEO = THRESHOLDS.video


def frame(index: int, image: Image, original_size: tuple[int, int] | None = None) -> SampledFrame:
    size = original_size or (image.shape[1], image.shape[0])
    return SampledFrame(index=index, time=index / FPS, image=image, original_size=size)


def detect(image: Image, channel: int = 0, label: str = "car") -> list[Detection]:
    box = find_box(image, channel)
    return [] if box is None else [Detection(box=box, label=label, score=0.9, detector="fake")]


def build(frames: list[SampledFrame], channel: int = 0) -> list[Track]:
    builder = TrackBuilder(VIDEO, FPS, THRESHOLDS.grouping.alpha)
    for f in frames:
        builder.update(f, detect(f.image, channel))
    return builder.tracks()


def moving(n: int, first: int = 0, colour: tuple[int, int, int] = RED, y: int = 30) -> list[
    SampledFrame
]:
    return [
        frame(first + i, frame_with_box((10 + 4 * i, y, 34 + 4 * i, y + 32), colour))
        for i in range(n)
    ]


def test_moving_box_becomes_one_track() -> None:
    (track,) = build(moving(12))
    assert track.id == 1
    assert track.label == "car"
    assert len(track.frames) >= 11  # ByteTrack may confirm a new track one frame late
    assert set(track.frames) == set(track.timestamps) == set(track.scores)
    assert track.status == "accepted"
    assert track.t_end > track.t_start


def test_two_objects_make_two_tracks() -> None:
    both = []
    for i in range(10):
        img = frame_with_box((10 + 3 * i, 10, 34 + 3 * i, 40), RED)
        img[55:85, 100:124] = BLUE
        both.append(frame(i, img))
    builder = TrackBuilder(VIDEO, FPS, THRESHOLDS.grouping.alpha)
    for f in both:
        builder.update(f, detect(f.image, 0) + detect(f.image, 2))
    assert len(builder.tracks()) == 2


def test_track_boxes_are_in_original_coordinates() -> None:
    # frames are working copies at half the original resolution
    frames = [frame(f.index, f.image, (320, 192)) for f in moving(8)]
    (track,) = build(frames)
    first = min(track.frames)
    working = find_box(frames[first].image)
    assert working is not None
    assert track.frames[first] == Box(*(2 * c for c in working))


def test_best_crops_are_ranked_and_capped() -> None:
    (track,) = build(moving(12))
    assert len(track.best_crops) == MAX_CROPS
    ranks = [c.rank for c in track.best_crops]
    assert ranks == sorted(ranks, reverse=True)
    assert track.best_crops[0].image.ndim == 3


def test_new_shot_starts_independent_tracks() -> None:
    builder = TrackBuilder(VIDEO, FPS, THRESHOLDS.grouping.alpha)
    for f in moving(6):
        builder.update(f, detect(f.image))
    builder.new_shot()
    for f in moving(6, first=6):
        builder.update(f, detect(f.image))
    assert len(builder.tracks()) == 2


def test_box_interpolation_between_samples() -> None:
    track = Track(
        id=1,
        frames={0: Box(0, 0, 10, 10), 5: Box(10, 20, 20, 30)},
        timestamps={0: 0.0, 5: 1.0},
        scores={0: 0.9, 5: 0.9},
        label="car",
    )
    at = BoxInterpolator(track)
    assert at(0.5) == Box(5, 10, 15, 20)
    assert at(0.0) == Box(0, 0, 10, 10)
    assert at(-0.1) is None
    assert at(1.2) is None


# ---- re-identification ------------------------------------------------------------------------


def broken_track_frames(gap: int, second: tuple[int, int, int] = RED) -> list[SampledFrame]:
    """A red box, `gap` empty frames, then a box of the `second` colour."""
    first = moving(6)
    empty = [frame(6 + i, background()) for i in range(gap)]
    return [*first, *empty, *moving(6, first=6 + gap, colour=second, y=34)]


def test_tracker_alone_fragments_a_track_after_a_long_gap() -> None:
    assert len(build(broken_track_frames(gap=5))) == 2


def test_reid_merges_a_track_broken_by_a_short_gap() -> None:
    tracks = build(broken_track_frames(gap=5))
    merged = merge_tracks(tracks, ColourEmbedder(), VIDEO)
    assert len(merged) == 1
    assert len(merged[0].frames) == sum(len(t.frames) for t in tracks)
    assert merged[0].t_start == tracks[0].t_start
    assert merged[0].t_end == tracks[1].t_end
    assert len(merged[0].best_crops) == MAX_CROPS


def test_reid_keeps_different_looking_objects_apart() -> None:
    # detect red and blue boxes with their own channel detectors
    builder = TrackBuilder(VIDEO, FPS, THRESHOLDS.grouping.alpha)
    for f in broken_track_frames(gap=5, second=BLUE):
        builder.update(f, detect(f.image, 0) + detect(f.image, 2))
    tracks = builder.tracks()
    assert len(tracks) == 2
    assert len(merge_tracks(tracks, ColourEmbedder(), VIDEO)) == 2


def test_reid_does_not_bridge_gaps_longer_than_the_maximum() -> None:
    tracks = build(broken_track_frames(gap=15))  # 3 s > reid_max_gap_seconds
    assert len(tracks) == 2
    assert len(merge_tracks(tracks, ColourEmbedder(), VIDEO)) == 2


def test_reid_requires_matching_labels() -> None:
    a, b = build(broken_track_frames(gap=5))
    b = b.model_copy(update={"label": "truck"})
    assert len(merge_tracks([a, b], ColourEmbedder(), VIDEO)) == 2


def test_reid_cosine_threshold_comes_from_config() -> None:
    a, b = build(broken_track_frames(gap=5))
    strict = settings_for(reid_cosine=1.01, reid_max_gap_seconds=2.0).thresholds.video
    assert len(merge_tracks([a, b], ColourEmbedder(), strict)) == 2


def test_reid_of_a_single_track_is_a_noop() -> None:
    (track,) = build(moving(6))
    assert merge_tracks([track], ColourEmbedder(), VIDEO) == [track]
    assert np.isfinite(track.mean_score)
