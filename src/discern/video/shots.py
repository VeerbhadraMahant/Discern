"""Shot boundaries (PySceneDetect content detector on sampled frames) and keyframes."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scenedetect import ContentDetector, FrameTimecode

from discern.config.settings import VideoThresholds
from discern.video.types import SampledFrame, Shot
from discern.vision.stats import laplacian_variance

# The keyframe is the sharpest sampled frame within this fraction of the shot length,
# centred on the shot midpoint.
KEYFRAME_WINDOW = 0.5


@dataclass(frozen=True)
class FrameScan:
    index: int  # original-video frame number
    time: float
    sharpness: float


def scan_frames(
    frames: Iterable[SampledFrame], sample_fps: float, thresholds: VideoThresholds
) -> tuple[list[FrameScan], list[int]]:
    """One pass over sampled frames: sharpness per frame and the scan positions where a
    new shot starts (cuts). Positions index into the returned scan list."""
    detector = ContentDetector(
        threshold=thresholds.cut_threshold,
        min_scene_len=max(1, round(thresholds.min_shot_seconds * sample_fps)),
    )
    scans: list[FrameScan] = []
    cuts: list[int] = []
    for position, frame in enumerate(frames):
        scans.append(FrameScan(frame.index, frame.time, laplacian_variance(frame.image)))
        bgr = np.ascontiguousarray(frame.image[..., ::-1])
        timecode = FrameTimecode(position, sample_fps)
        cuts.extend(c.frame_num for c in detector.process_frame(timecode, bgr))
    if scans:
        last = FrameTimecode(len(scans), sample_fps)
        cuts.extend(c.frame_num for c in detector.post_process(last))
    return scans, sorted({c for c in cuts if 0 < c < len(scans)})


def build_shots(scans: Sequence[FrameScan], cuts: Sequence[int], end_time: float) -> list[Shot]:
    """Split scans at the cut positions. No cuts (or a single frame) gives one shot.
    The keyframe is the sharpest scan near the shot midpoint."""
    if not scans:
        return []
    bounds = [0, *cuts, len(scans)]
    shots: list[Shot] = []
    for shot_id, (lo, hi) in enumerate(zip(bounds, bounds[1:], strict=False)):
        members = scans[lo:hi]
        t_start = members[0].time
        t_end = scans[hi].time if hi < len(scans) else max(end_time, members[-1].time)
        key = _keyframe(members, t_start, t_end)
        shots.append(
            Shot(
                id=shot_id,
                t_start=t_start,
                t_end=t_end,
                keyframe_index=key.index,
                keyframe_time=key.time,
            )
        )
    return shots


def _keyframe(members: Sequence[FrameScan], t_start: float, t_end: float) -> FrameScan:
    mid = (t_start + t_end) / 2.0
    half = KEYFRAME_WINDOW * (t_end - t_start) / 2.0
    near = [m for m in members if abs(m.time - mid) <= half]
    pool = near or [min(members, key=lambda m: abs(m.time - mid))]
    return max(pool, key=lambda m: (m.sharpness, -abs(m.time - mid)))


def detect_shots(
    frames: Iterable[SampledFrame],
    sample_fps: float,
    end_time: float,
    thresholds: VideoThresholds,
) -> list[Shot]:
    scans, cuts = scan_frames(frames, sample_fps, thresholds)
    return build_shots(scans, cuts, end_time)
