"""Probe and decode video with PyAV. File names are only ever passed to the decoder,
never interpolated into a shell command."""

from collections.abc import Iterator
from pathlib import Path

import av
import av.error
import av.video.frame

from discern.config.settings import Profile
from discern.video.types import SampledFrame, VideoError, VideoInfo


def open_video(path: Path) -> av.container.InputContainer:
    if not path.is_file():
        raise VideoError(f"video file not found: {path.name}")
    try:
        return av.open(str(path))
    except (av.error.FFmpegError, OSError) as err:
        raise VideoError(f"cannot open {path.name} as video: {err}") from err


def probe(path: Path) -> VideoInfo:
    """Read fps, duration, size and frame count from the container headers."""
    with open_video(path) as container:
        if not container.streams.video:
            raise VideoError(f"{path.name} has no video stream")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if rate is None or float(rate) <= 0:
            raise VideoError(f"{path.name} has no usable frame rate")
        fps = float(rate)
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base
        else:
            duration = stream.frames / fps
        frames = stream.frames or round(duration * fps)
        width, height = stream.codec_context.width, stream.codec_context.height
    if width <= 0 or height <= 0 or duration <= 0:
        raise VideoError(f"{path.name} reports an empty video")
    return VideoInfo(fps=fps, duration=duration, width=width, height=height, frame_count=frames)


def validate(info: VideoInfo, profile: Profile) -> None:
    """Reject videos longer than the profile allows. Resolution is not a limit: frames are
    downscaled to `max_long_side_px` at decode time instead."""
    if info.duration > profile.max_video_seconds:
        raise VideoError(
            f"video is {info.duration:.1f}s long; profile {profile.name!r} allows at most "
            f"{profile.max_video_seconds}s"
        )


def working_size(width: int, height: int, max_long_side: int) -> tuple[int, int]:
    """Downscale so the long side is at most max_long_side; never upscale."""
    factor = min(1.0, max_long_side / max(width, height))
    return max(1, round(width * factor)), max(1, round(height * factor))


def iter_sampled_frames(
    path: Path, sample_fps: float, max_long_side: int
) -> Iterator[SampledFrame]:
    """Yield frames about every 1/sample_fps seconds (all frames if the video is slower),
    as RGB uint8 arrays downscaled to the working resolution."""
    with open_video(path) as container:
        if not container.streams.video:
            raise VideoError(f"{path.name} has no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fallback_fps = float(stream.average_rate or stream.guessed_rate or 1)
        next_time = 0.0
        step = 1.0 / sample_fps
        size: tuple[int, int] | None = None
        try:
            for index, frame in enumerate(container.decode(stream)):
                t = float(frame.time) if frame.time is not None else index / fallback_fps
                if t + 1e-6 < next_time:
                    continue
                next_time += step
                if next_time <= t:  # a timestamp gap or offset: restart the grid at this frame
                    next_time = t + step
                original = (frame.width, frame.height)
                size = size or working_size(*original, max_long_side)
                rgb = frame.reformat(width=size[0], height=size[1], format="rgb24")
                yield SampledFrame(
                    index=index, time=t, image=rgb.to_ndarray(), original_size=original
                )
        except av.error.FFmpegError as err:
            raise VideoError(f"decoding {path.name} failed: {err}") from err
