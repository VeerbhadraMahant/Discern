"""Synthetic videos for tests: moving coloured rectangles on a textured background."""

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from discern.config.settings import Settings, load_settings
from discern.models.roles import Embeddings, Image
from discern.vision.boxes import Box

SIZE = (160, 96)  # (width, height); even, as yuv420p requires
RED = (220, 40, 40)
BLUE = (40, 60, 220)


def background(seed: int = 0, level: int = 90, size: tuple[int, int] = SIZE) -> Image:
    noise = np.random.default_rng(seed).integers(-12, 13, size=(size[1], size[0], 1))
    return np.clip(level + noise, 0, 255).astype(np.uint8).repeat(3, axis=2)


def frame_with_box(
    box: tuple[int, int, int, int],
    colour: tuple[int, int, int] = RED,
    bg: Image | None = None,
) -> Image:
    img = (background() if bg is None else bg).copy()
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = colour
    return img


def moving_box_frames(
    n: int, start_x: int = 10, step: int = 4, y: int = 30, w: int = 24, h: int = 32,
    colour: tuple[int, int, int] = RED, bg: Image | None = None,
) -> list[Image]:
    return [frame_with_box((start_x + i * step, y, start_x + i * step + w, y + h), colour, bg)
            for i in range(n)]


def write_clip(
    path: Path,
    frames: Sequence[Image],
    fps: int = 10,
    times_ms: Sequence[int] | None = None,
    pix_fmt: str = "yuv420p",
) -> Path:
    """Encode `frames` as H.264. `times_ms` gives each frame's presentation time (variable frame
    rate, or a start offset); `pix_fmt="yuv444p"` allows odd dimensions."""
    height, width = frames[0].shape[:2]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps, 1))
        stream.width, stream.height, stream.pix_fmt = width, height, pix_fmt
        stream.options = {"crf": "10", "preset": "ultrafast"}
        if times_ms is not None:
            stream.time_base = Fraction(1, 1000)
        for i, img in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            if times_ms is not None:
                frame.pts, frame.time_base = times_ms[i], Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def settings_for(sample_fps: float = 5.0, max_long_side: int = 1280, **video: float) -> Settings:
    """Settings with test sampling and optional video-threshold overrides."""
    base = load_settings("local_lite")
    profile = base.profile.model_copy(
        update={
            "sample_fps": sample_fps,
            "max_long_side_px": max_long_side,
            "max_video_seconds": 60,
        }
    )
    thresholds = base.thresholds.model_copy(
        update={"video": base.thresholds.video.model_copy(update=video)}
    )
    return Settings(profile=profile, thresholds=thresholds)


def find_box(image: Image, channel: int = 0, margin: int = 60) -> Box | None:
    """Bounding box of pixels whose `channel` exceeds the mean of the other channels by margin."""
    rgb = image.astype(np.int16)
    others = (rgb.sum(axis=2) - rgb[..., channel]) / 2
    ys, xs = np.nonzero(rgb[..., channel] - others > margin)
    if len(xs) == 0:
        return None
    return Box(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


class ColourEmbedder:
    """One-hot of the crop's strongest colour channel: same-coloured crops are identical."""

    def embed_images(self, images: Sequence[Image]) -> Embeddings:
        out = np.zeros((len(images), 3), dtype=np.float32)
        for i, img in enumerate(images):
            out[i, int(img.reshape(-1, 3).mean(axis=0).argmax())] = 1.0
        return out

    def embed_text(self, texts: Sequence[str]) -> Embeddings:
        raise NotImplementedError
