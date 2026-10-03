"""Tracking: ByteTrack (supervision) over per-frame fused detections, box interpolation for
rendering, and embedding-based re-identification of fragmented tracks."""

import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import supervision as sv

from discern.config.settings import VideoThresholds
from discern.models.roles import Detection, Embedder
from discern.video.types import SampledFrame, Track, TrackCrop
from discern.vision.boxes import Box
from discern.vision.grouping import crop_box
from discern.vision.stats import laplacian_variance

MAX_CROPS = 3  # best crops kept per track (system-design 5.2 item 8)
SUPERVISION_FPS_REFERENCE = 30  # lost_track_buffer is expressed in frames at this frame rate


def _best_first(crops: Sequence[TrackCrop]) -> list[TrackCrop]:
    return sorted(crops, key=lambda c: -c.rank)[:MAX_CROPS]


@dataclass
class _Observations:
    frames: dict[int, Box] = field(default_factory=dict)
    timestamps: dict[int, float] = field(default_factory=dict)
    scores: dict[int, float] = field(default_factory=dict)
    labels: Counter[str] = field(default_factory=Counter)
    crops: list[TrackCrop] = field(default_factory=list)


class TrackBuilder:
    """Feed sampled frames in order with their fused detections (working-resolution boxes);
    `tracks()` returns tracks with boxes mapped back to original-frame coordinates."""

    def __init__(self, thresholds: VideoThresholds, sample_fps: float, crop_alpha: float) -> None:
        self._thresholds = thresholds
        self._fps = sample_fps
        self._alpha = crop_alpha
        self._observations: dict[tuple[int, int], _Observations] = {}
        self._shot = 0
        self._tracker = self._new_tracker()

    def _new_tracker(self) -> sv.ByteTrack:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # ByteTrack deprecated, kept to 0.30
            return sv.ByteTrack(
                track_activation_threshold=self._thresholds.track_activation_score,
                lost_track_buffer=round(
                    self._thresholds.track_lost_seconds * SUPERVISION_FPS_REFERENCE
                ),
                frame_rate=max(1, round(self._fps)),
            )

    def new_shot(self) -> None:
        """A cut ends every open track; ids in the next shot are independent."""
        self._shot += 1
        self._tracker = self._new_tracker()

    def update(self, frame: SampledFrame, detections: Sequence[Detection]) -> None:
        if not detections:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                self._tracker.update_with_detections(sv.Detections.empty())
            return
        xyxy: npt.NDArray[np.float32] = np.array([tuple(d.box) for d in detections], np.float32)
        found = sv.Detections(
            xyxy=xyxy,
            confidence=np.array([d.score for d in detections], np.float32),
            class_id=np.arange(len(detections)),  # carries the detection index through
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            tracked = self._tracker.update_with_detections(found)
        if tracked.tracker_id is None or tracked.class_id is None:
            return
        for tracker_id, index in zip(tracked.tracker_id, tracked.class_id, strict=True):
            det = detections[int(index)]
            obs = self._observations.setdefault((self._shot, int(tracker_id)), _Observations())
            original = frame.to_original(det.box)
            obs.frames[frame.index] = original
            obs.timestamps[frame.index] = frame.time
            obs.scores[frame.index] = det.score
            obs.labels[det.label] += 1
            crop = crop_box(frame.image, det.box, self._alpha)
            rank = det.box.area * det.score * laplacian_variance(crop)
            obs.crops = _best_first(
                [*obs.crops, TrackCrop(
                    frame_index=frame.index, time=frame.time, box=original, rank=rank,
                    image=crop.copy(),
                )]
            )

    def tracks(self) -> list[Track]:
        ordered = sorted(
            self._observations.values(), key=lambda o: min(o.timestamps.values())
        )
        return [
            Track(
                id=i,
                frames=o.frames,
                timestamps=o.timestamps,
                scores=o.scores,
                label=o.labels.most_common(1)[0][0],
                best_crops=o.crops,
            )
            for i, o in enumerate(ordered, start=1)
        ]


class BoxInterpolator:
    """Linear box interpolation between a track's sampled frames, for rendering."""

    def __init__(self, track: Track) -> None:
        indices = sorted(track.frames, key=lambda i: track.timestamps[i])
        self._times = np.array([track.timestamps[i] for i in indices])
        self._boxes = np.array([tuple(track.frames[i]) for i in indices])

    def __call__(self, t: float) -> Box | None:
        """The box at time t, or None outside the track's observed time span."""
        if t < self._times[0] - 1e-9 or t > self._times[-1] + 1e-9:
            return None
        x1, y1, x2, y2 = (float(np.interp(t, self._times, self._boxes[:, k])) for k in range(4))
        return Box(x1, y1, x2, y2)


def _track_embedding(embeddings: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    mean = embeddings.astype(np.float64).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return np.asarray(mean / norm if norm > 0 else mean)


def merge_tracks(
    tracks: Sequence[Track], embedder: Embedder, thresholds: VideoThresholds
) -> list[Track]:
    """Merge a track into an earlier one when the labels match, the gap between them is at most
    `reid_max_gap_seconds`, they do not overlap in time, and the cosine similarity of their
    mean crop embeddings is at least `reid_cosine`. The best earlier candidate wins."""
    ordered = sorted(tracks, key=lambda t: t.t_start)
    if len(ordered) < 2:
        return list(ordered)
    crops = [c.image for t in ordered for c in t.best_crops]
    all_embeddings = embedder.embed_images(crops)
    chains: list[Track] = []
    chain_embeddings: list[npt.NDArray[np.float64]] = []
    offset = 0
    for track in ordered:
        n = len(track.best_crops)
        emb = _track_embedding(all_embeddings[offset : offset + n])
        offset += n
        best, best_cos = -1, thresholds.reid_cosine
        for i, chain in enumerate(chains):
            gap = track.t_start - chain.t_end
            if chain.label != track.label or gap <= 0 or gap > thresholds.reid_max_gap_seconds:
                continue
            cos = float(np.dot(chain_embeddings[i], emb))
            if cos >= best_cos:
                best, best_cos = i, cos
        if best < 0:
            chains.append(track)
            chain_embeddings.append(emb)
            continue
        head = chains[best]
        chains[best] = head.model_copy(
            update={
                "frames": {**head.frames, **track.frames},
                "timestamps": {**head.timestamps, **track.timestamps},
                "scores": {**head.scores, **track.scores},
                "best_crops": _best_first([*head.best_crops, *track.best_crops]),
            }
        )
        chain_embeddings[best] = _track_embedding(np.stack([chain_embeddings[best], emb]))
    return chains
