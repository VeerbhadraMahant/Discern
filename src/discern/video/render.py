"""Annotated video output (PyAV, PIL drawing; no OpenCV) and per-track JSON export.

Encoding prefers H.264 (libx264) and falls back to MPEG-4 part 2 (mpeg4) when the PyAV build
has no H.264 encoder. Width and height are cut down to even numbers, which yuv420p requires.
"""

import json
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from av.video.stream import VideoStream
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from discern.video.io import open_video
from discern.video.tracks import BoxInterpolator
from discern.video.types import Track

COLOURS = (
    (230, 60, 60),
    (60, 200, 90),
    (70, 110, 240),
    (240, 200, 50),
    (220, 80, 220),
    (60, 210, 220),
)
FONT_HEIGHT_DIVISOR = 45  # font pixel size = frame height / divisor
MIN_FONT_PX = 10
LINE_HEIGHT_DIVISOR = 240
MIN_LINE_PX = 2
MAX_RATE_DENOMINATOR = 1001  # keeps 29.97 fps as 30000/1001


def _add_video_stream(out: av.container.OutputContainer, rate: Fraction) -> VideoStream:
    """H.264 if this PyAV build has it, else MPEG-4 part 2."""
    try:
        return out.add_stream("libx264", rate=rate)
    except (av.error.FFmpegError, ValueError):
        return out.add_stream("mpeg4", rate=rate)


def render_video(src: Path, tracks: Sequence[Track], dst: Path) -> int:
    """Draw accepted tracks (box, label, id) and the timestamp on every frame of `src` at the
    original resolution and frame rate. Boxes between sampled frames are interpolated.
    Returns the number of frames written."""
    shown = [(t, BoxInterpolator(t)) for t in tracks if t.status == "accepted"]
    written = 0
    with open_video(src) as source, av.open(str(dst), "w") as out:
        in_stream = source.streams.video[0]
        native = in_stream.average_rate or in_stream.guessed_rate or 25
        rate = Fraction(float(native)).limit_denominator(MAX_RATE_DENOMINATOR)
        width, height = in_stream.codec_context.width & ~1, in_stream.codec_context.height & ~1
        stream = _add_video_stream(out, rate)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        font = ImageFont.load_default(size=max(MIN_FONT_PX, height // FONT_HEIGHT_DIVISOR))
        line = max(MIN_LINE_PX, height // LINE_HEIGHT_DIVISOR)
        for index, frame in enumerate(source.decode(in_stream)):
            t = float(frame.time) if frame.time is not None else index / float(rate)
            pil = PILImage.fromarray(frame.to_ndarray(format="rgb24")[:height, :width])
            draw = ImageDraw.Draw(pil)
            for track, interpolate in shown:
                box = interpolate(t)
                if box is None:
                    continue
                colour = COLOURS[track.id % len(COLOURS)]
                draw.rectangle(tuple(box), outline=colour, width=line)
                draw.text(
                    (box.x1 + line, box.y1 + line),
                    f"#{track.id} {track.label}",
                    fill=colour,
                    font=font,
                )
            draw.text((line, line), f"t={t:.2f}s", fill=(255, 255, 255), font=font)
            video_frame = av.VideoFrame.from_ndarray(np.asarray(pil), format="rgb24")
            for packet in stream.encode(video_frame):
                out.mux(packet)
            written += 1
        for packet in stream.encode():
            out.mux(packet)
    return written


def tracks_to_json(tracks: Sequence[Track]) -> str:
    """Per-track JSON: label, status, rationale, and per-frame boxes, times and scores."""
    payload = [
        {
            **t.model_dump(mode="json", exclude={"best_crops"}),
            "t_start": t.t_start,
            "t_end": t.t_end,
        }
        for t in tracks
    ]
    return json.dumps(payload, indent=2)
