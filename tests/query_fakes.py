"""Fakes and builders for the index and query tests: tracks from boxes, a synthetic index, a
scripted text embedder and a track provider that counts its calls."""

from collections.abc import Sequence

import numpy as np

from discern.config.settings import load_settings
from discern.index.video_index import Segment, VideoIndex
from discern.models.fakes import FakeVLM
from discern.models.roles import Embeddings, Image
from discern.query.executors import Services
from discern.trace import TraceCollector
from discern.video.types import Shot, Track
from discern.vision.boxes import Box

DT = 0.5  # seconds between sampled frames
FRAME_SIZE = (200, 100)

Rect = tuple[float, float, float, float]


def make_track(
    track_id: int,
    label: str,
    boxes: Sequence[Rect],
    t0: float = 0.0,
    status: str = "accepted",
    score: float = 0.9,
) -> Track:
    """A track with one box per sampled frame, starting at time t0. Frame index = time / DT."""
    first = round(t0 / DT)
    indices = [first + i for i in range(len(boxes))]
    return Track(
        id=track_id,
        frames={i: Box(*b) for i, b in zip(indices, boxes, strict=True)},
        timestamps={i: i * DT for i in indices},
        scores={i: score for i in indices},
        label=label,
        status="accepted" if status == "accepted" else "rejected",
    )


def still(rect: Rect, n: int) -> list[Rect]:
    return [rect] * n


class ScriptedEmbedder:
    """Text embeddings from a lookup (default [1, 0]); images are not embedded in these tests."""

    def __init__(self, texts: dict[str, Sequence[float]]) -> None:
        self._texts = texts

    def embed_images(self, images: Sequence[Image]) -> Embeddings:
        raise NotImplementedError

    def embed_text(self, texts: Sequence[str]) -> Embeddings:
        return np.array([self._texts.get(t, (1.0, 0.0)) for t in texts], dtype=np.float32)


def make_index(n: int = 20, captions: dict[int, str] | None = None) -> VideoIndex:
    """n frames at 1 s, all embedded as [0, 1]; two shots split at n / 2."""
    half = n // 2
    shots = [
        Shot(id=0, t_start=0.0, t_end=float(half), keyframe_index=0, keyframe_time=0.0),
        Shot(id=1, t_start=float(half), t_end=float(n), keyframe_index=half, keyframe_time=half),
    ]
    return VideoIndex(
        times=np.arange(n, dtype=np.float64),
        embeddings=np.tile(np.array([0.0, 1.0], np.float32), (n, 1)),
        shots=shots,
        captions=captions if captions is not None else {0: "a road", 1: "a parking lot"},
    )


class SpyProvider:
    """A TrackProvider returning scripted tracks per target and recording every call."""

    def __init__(self, tracks: dict[str, list[Track]], empty_on_segments: bool = False) -> None:
        self._tracks = tracks
        self._empty_on_segments = empty_on_segments
        self.calls: list[tuple[tuple[str, ...], bool]] = []  # (targets, had segments)

    def __call__(
        self, targets: Sequence[str], segments: Sequence[Segment] | None
    ) -> list[Track]:
        self.calls.append((tuple(targets), segments is not None))
        if segments is not None and self._empty_on_segments:
            return []
        return [t for target in targets for t in self._tracks.get(target, [])]


def make_services(
    vlm: FakeVLM,
    provider: SpyProvider,
    trace: TraceCollector | None = None,
    index: VideoIndex | None = None,
) -> Services:
    return Services(
        index=index or make_index(),
        embedder=ScriptedEmbedder({}),
        vlm=vlm,
        trace=trace or TraceCollector(),
        settings=load_settings("local_lite"),
        frame_size=FRAME_SIZE,
        detect=provider,
    )
