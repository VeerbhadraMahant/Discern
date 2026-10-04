"""Video orchestration: ingest (decode, shots, per-shot SAIR plan) and tracking (detect,
track, re-identify, adjudicate). Detection is supplied by the caller as a callable."""

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from discern.agent.nodes.adjudicate_track import adjudicate_track
from discern.agent.nodes.restorer_select import NONE
from discern.agent.sair import Experience, Policy, plan_image
from discern.agent.schemas import ShotPlan
from discern.config.settings import Settings, load_settings
from discern.models.roles import VLM, Detection, Embedder, Image, Restorer
from discern.trace import TraceCollector
from discern.video.io import decode_limit, iter_sampled_frames, probe, validate
from discern.video.shots import detect_shots
from discern.video.tracks import TrackBuilder, merge_tracks
from discern.video.types import SampledFrame, Shot, Track, VideoError, VideoInfo

logger = logging.getLogger(__name__)

# Per-frame detection: (shot, working-resolution RGB frame) -> fused detections in that frame's
# pixel coordinates (for example multi-detector proposals fused with `group_detections`).
FrameDetector = Callable[[Shot, Image], list[Detection]]

# Time windows (start, end) in seconds, both inclusive, that limit which frames are processed.
Windows = Sequence[tuple[float, float]]


def _window_index(t: float, windows: Windows) -> int | None:
    """Index of the first window containing time `t`, or None."""
    return next((i for i, (start, end) in enumerate(windows) if start <= t <= end), None)


@dataclass
class VideoIngest:
    path: Path
    info: VideoInfo
    settings: Settings
    shots: list[Shot]
    plans: dict[int, ShotPlan]  # shot id -> cached SAIR plan
    restorers: Mapping[str, Restorer]

    def shot_at(self, t: float) -> Shot:
        eligible = [s for s in self.shots if s.t_start <= t + 1e-6]
        return eligible[-1] if eligible else self.shots[0]

    def frames(self, windows: Windows | None = None) -> Iterator[tuple[Shot, SampledFrame]]:
        """Re-decode the sampled frames, applying each shot's plan with the same restorer and
        parameters to every frame of that shot. Frames are not cached in memory. With `windows`,
        only frames inside them are restored and yielded, and decoding stops after the last one."""
        profile = self.settings.profile
        last = max((end for _, end in windows), default=-1.0) if windows is not None else None
        for frame in iter_sampled_frames(
            self.path, profile.sample_fps, profile.max_long_side_px, decode_limit(self.settings)
        ):
            if last is not None and frame.time > last:
                break
            if windows is not None and _window_index(frame.time, windows) is None:
                continue
            shot = self.shot_at(frame.time)
            plan = self.plans[shot.id]
            if plan.use_restored and plan.restorer != NONE:
                try:
                    restored = self.restorers[plan.restorer].restore(frame.image)
                except Exception:
                    logger.exception("restorer %s failed on frame %d", plan.restorer, frame.index)
                else:
                    frame = frame.model_copy(update={"image": restored})
            yield shot, frame


def ingest_video(
    path: Path,
    vlm: VLM,
    trace: TraceCollector,
    restorers: Mapping[str, Restorer],
    experience: Experience = "",
    settings: Settings | None = None,
    policy: Policy = None,
) -> VideoIngest:
    """Probe and validate, split into shots, and plan SAIR once per shot on its keyframe.

    Super-resolution is decided by `plan_image` but never applied to video (system-design
    section 12: off by default in video).
    """
    settings = settings or load_settings()
    profile = settings.profile
    info = probe(path)
    validate(info, profile)
    with trace.span("video_ingest") as span:
        span.input_summary = (
            f"{info.width}x{info.height}, {info.fps:.2f} fps, {info.duration:.1f}s, "
            f"sample_fps={profile.sample_fps}"
        )
        shots = detect_shots(
            iter_sampled_frames(
                path, profile.sample_fps, profile.max_long_side_px, decode_limit(settings)
            ),
            profile.sample_fps,
            info.duration,
            settings.thresholds.video,
        )
        if not shots:
            raise VideoError(f"{path.name} has no decodable frames")
        wanted = {s.keyframe_index: s.id for s in shots}
        keyframes: dict[int, Image] = {}
        for frame in iter_sampled_frames(
            path, profile.sample_fps, profile.max_long_side_px, decode_limit(settings)
        ):
            if frame.index in wanted:
                keyframes[wanted[frame.index]] = frame.image
                if len(keyframes) == len(wanted):
                    break
        plans = {
            shot.id: plan_image(
                vlm, trace, keyframes[shot.id], restorers, experience, settings, policy
            )[0]
            for shot in shots
        }
        span.decision = f"{len(shots)} shots; restorers: " + ", ".join(
            f"{s.id}={plans[s.id].restorer if plans[s.id].use_restored else NONE}" for s in shots
        )
    return VideoIngest(path, info, settings, shots, plans, restorers)


def track_video(
    ingest: VideoIngest,
    detect: FrameDetector,
    embedder: Embedder,
    vlm: VLM,
    trace: TraceCollector,
    targets: Sequence[str],
    windows: Windows | None = None,
) -> list[Track]:
    """Detect on every sampled frame (only those inside `windows` when given; tracks never span
    two windows), track, re-identify, then adjudicate each track once.
    Rejected tracks stay in the result with status "rejected"."""
    settings = ingest.settings
    thresholds = settings.thresholds
    builder = TrackBuilder(thresholds.video, settings.profile.sample_fps, thresholds.grouping.alpha)
    with trace.span("track") as span:
        span.input_summary = f"targets={list(targets)}, shots={len(ingest.shots)}"
        current: tuple[int, int | None] | None = None
        for shot, frame in ingest.frames(windows):
            key = (shot.id, None if windows is None else _window_index(frame.time, windows))
            if current is not None and key != current:
                builder.new_shot()
            current = key
            builder.update(frame, detect(shot, frame.image))
        raw = builder.tracks()
        span.decision = f"{len(raw)} tracks"
    with trace.span("reid") as span:
        span.input_summary = f"tracks={len(raw)}"
        merged = merge_tracks(raw, embedder, thresholds.video)
        span.decision = f"{len(raw)} -> {len(merged)} tracks"
    result: list[Track] = []
    for track in merged:
        verdict = adjudicate_track(vlm, trace, track, targets, settings)
        result.append(
            track.model_copy(
                update={
                    "status": "accepted" if verdict.accept else "rejected",
                    "label": verdict.label if verdict.accept and verdict.label else track.label,
                    "rationale": verdict.rationale,
                }
            )
        )
    return result
