"""Video index and retrieval (system-design 5.4): frame embeddings with timestamps, one caption
per shot, and text-to-segment retrieval. Retrieval is a pruning aid, not ground truth."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from discern.agent.nodes.caption import caption
from discern.config.settings import Settings, load_settings
from discern.models.roles import VLM, Embedder, Embeddings, Image
from discern.trace import TraceCollector
from discern.video.pipeline import VideoIngest
from discern.video.types import Shot

EMBED_BATCH = 32  # frames embedded per call
ARRAYS_FILE = "index.npz"
META_FILE = "index.json"


class Segment(BaseModel):
    t_start: float
    t_end: float
    score: float  # smoothed cosine similarity at the segment's peak


@dataclass
class VideoIndex:
    times: npt.NDArray[np.float64]  # sampled frame timestamps, ascending
    embeddings: Embeddings  # one L2-normalised row per timestamp
    shots: list[Shot]
    captions: dict[int, str]  # shot id -> caption; "" when none could be produced

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / ARRAYS_FILE, times=self.times, embeddings=self.embeddings)
        meta = {
            "shots": [s.model_dump() for s in self.shots],
            "captions": {str(k): v for k, v in self.captions.items()},
        }
        (directory / META_FILE).write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VideoIndex":
        with np.load(directory / ARRAYS_FILE) as arrays:
            times, embeddings = arrays["times"], arrays["embeddings"]
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        return cls(
            times=times,
            embeddings=embeddings,
            shots=[Shot.model_validate(s) for s in meta["shots"]],
            captions={int(k): v for k, v in meta["captions"].items()},
        )


def _normalise(rows: Embeddings) -> Embeddings:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.asarray(rows / np.where(norms > 0, norms, 1.0), dtype=np.float32)


def build_index(
    ingest: VideoIngest, embedder: Embedder, vlm: VLM, trace: TraceCollector
) -> VideoIndex:
    """Embed every sampled frame (after each shot's restoration plan) and caption each shot's
    keyframe. One pass over the video; only keyframes are kept in memory."""
    keyframes: dict[int, Image] = {}
    wanted = {s.keyframe_index: s.id for s in ingest.shots}
    times: list[float] = []
    chunks: list[Embeddings] = []
    batch: list[Image] = []
    with trace.span("index_build") as span:
        span.input_summary = f"shots={len(ingest.shots)}"
        for _, frame in ingest.frames():
            times.append(frame.time)
            batch.append(frame.image)
            if frame.index in wanted:
                keyframes[wanted[frame.index]] = frame.image.copy()
            if len(batch) == EMBED_BATCH:
                chunks.append(embedder.embed_images(batch))
                batch = []
        if batch:
            chunks.append(embedder.embed_images(batch))
        captions = {
            shot.id: caption(vlm, trace, keyframes[shot.id]) if shot.id in keyframes else ""
            for shot in ingest.shots
        }
        span.decision = f"{len(times)} frames, {sum(1 for c in captions.values() if c)} captions"
    embeddings = _normalise(np.concatenate(chunks)) if chunks else np.zeros((0, 0), np.float32)
    return VideoIndex(np.array(times, dtype=np.float64), embeddings, ingest.shots, captions)


def retrieve(
    index: VideoIndex,
    embedder: Embedder,
    text: str,
    top_k: int,
    settings: Settings | None = None,
) -> list[Segment]:
    """Top `top_k` segments for a text query, best first.

    Frames are scored by cosine similarity, averaged over a centred window of
    `index.smoothing_seconds`, and the best peaks (at least one window apart) each become a
    segment one window long, clipped to the video.
    """
    if len(index.times) == 0:
        return []
    window = (settings or load_settings()).thresholds.index.smoothing_seconds
    half = window / 2
    query = _normalise(embedder.embed_text([text]))[0]
    sims = index.embeddings @ query
    cumulative = np.concatenate([[0.0], np.cumsum(sims, dtype=np.float64)])
    lo = np.searchsorted(index.times, index.times - half, side="left")
    hi = np.searchsorted(index.times, index.times + half, side="right")
    smoothed = (cumulative[hi] - cumulative[lo]) / (hi - lo)
    peaks: list[int] = []
    for i in np.argsort(-smoothed, kind="stable"):
        if all(abs(index.times[i] - index.times[p]) >= window for p in peaks):
            peaks.append(int(i))
            if len(peaks) == top_k:
                break
    return [
        Segment(
            t_start=float(max(index.times[0], index.times[p] - half)),
            t_end=float(min(index.times[-1], index.times[p] + half)),
            score=float(smoothed[p]),
        )
        for p in peaks
    ]

