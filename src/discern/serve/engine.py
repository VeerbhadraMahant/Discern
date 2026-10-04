"""The serving engine: everything the UI calls, with no gradio import (system-design 8).

Models are built lazily through `ModelManager` and `load_adapter`; tests inject a `loader` that
returns fakes. GPU-bound work lives in private `_..._gpu` methods wrapped by `gpu(...)`. Those
return their results and trace events instead of mutating caller state, so the same code is correct
whether the wrapped call runs in this process or in a ZeroGPU worker.
"""

import gc
import hashlib
import logging
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from discern.agent.med import detect_image
from discern.agent.nodes.restorer_select import NONE
from discern.agent.sair import plan_image
from discern.agent.schemas import DetectorInfo, SceneProfile, ShotPlan
from discern.config.settings import ServeThresholds, Settings, load_settings
from discern.experience.aggregate import Memory, MemoryVersion, load_memory, memory_path
from discern.experience.injection import render_for
from discern.experience.policy import (
    DecisionPolicy,
    build_policy,
    detector_decision,
    record_decision,
)
from discern.experience.promotion import read_pointer
from discern.experience.schema import Node
from discern.feedback.schema import Feedback, FeedbackStore, StoredFeedback
from discern.index.video_index import Segment, VideoIndex, build_index
from discern.models.loading import load_adapter
from discern.models.manager import ModelManager, RegistryEntry
from discern.models.registry import load_registry
from discern.models.roles import SCORE_FLOOR, Detection, Image
from discern.monitor.summary import SessionRecord, render_markdown, summarise
from discern.query.executors import Services
from discern.query.schemas import ConversationState, ResultSet
from discern.query.session import answer_question
from discern.serve.gpu import gpu, gpu_available
from discern.serve.session import SessionStore, validate_upload
from discern.trace import TraceCollector, TraceEvent
from discern.video.io import decode_limit, iter_sampled_frames, working_size
from discern.video.pipeline import VideoIngest, ingest_video, track_video
from discern.video.render import render_video
from discern.video.types import Shot, Track, VideoInfo
from discern.vision.boxes import Box, scale
from discern.vision.grouping import group_detections

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]
Loader = Callable[[RegistryEntry], object]
SESSIONS_ENV_VAR = "DISCERN_SESSIONS_DIR"
FEEDBACK_ENV_VAR = "DISCERN_FEEDBACK_DIR"
BOX_COLOUR = (230, 60, 60)
LABEL_COLOUR = (255, 255, 255)
MIN_LINE_PX = 2
LINE_DIVISOR = 300  # box line width = long side / divisor
_NON_COMMERCIAL_MARKERS = ("nc", "non-commercial", "academic")
_SECONDS_PER_MINUTE = 60.0


class EngineError(ValueError):
    """A request the engine refuses; the message is safe to show the user."""


class LimitError(EngineError):
    """A configured limit was reached."""


@dataclass
class CleanResult:
    original: Image  # full-resolution RGB as uploaded
    working: Image  # the downscaled copy the agent saw (same object as original if not downscaled)
    cleaned: Image  # restored (if chosen) and super-resolved (if chosen) working copy
    plan: ShotPlan
    profile: SceneProfile
    events: list[TraceEvent]
    session_id: str | None = None

    def to_original(self, box: Box) -> Box:
        """Map a box found on `cleaned` back into the uploaded image's pixel space."""
        return scale(
            box,
            self.original.shape[1] / self.cleaned.shape[1],
            self.original.shape[0] / self.cleaned.shape[0],
        )


@dataclass
class DetectResult:
    annotated: Image  # the uploaded image with boxes drawn
    detections: list[Detection]  # boxes in the uploaded image's pixel space
    events: list[TraceEvent]


@dataclass
class TrackEvidence:
    track_id: int
    label: str
    t_start: float
    t_end: float
    crop: Image | None


@dataclass
class Evidence:
    result_set_id: str | None
    grounded: bool
    count: int | None
    tracks: list[TrackEvidence]
    events: list[TraceEvent]


@dataclass
class VideoSession:
    session_id: str
    ingest: VideoIngest
    index: VideoIndex
    conversation: ConversationState = field(default_factory=ConversationState)

    @property
    def queries_asked(self) -> int:
        return len(self.conversation.turns)


@dataclass(frozen=True)
class RequestTrace:
    kind: str
    events: tuple[TraceEvent, ...]


class LazyRoles(Mapping[str, Any]):
    """Models of the profile roles with one prefix (for example `restorer_`), keyed by short name
    and loaded through the `ModelManager` on first use. Membership never loads a model."""

    def __init__(
        self, manager: Callable[[], ModelManager], roles: Mapping[str, tuple[str, str]]
    ) -> None:
        self._manager = manager
        self._roles = dict(roles)  # key -> (role, registry entry name)

    def __getitem__(self, key: str) -> Any:
        role, name = self._roles[key]
        try:
            return self._manager().get(role, name)
        except Exception:
            logger.exception("loading %s failed", name)
            raise KeyError(key) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._roles)

    def __len__(self) -> int:
        return len(self._roles)

    def __contains__(self, key: object) -> bool:
        return key in self._roles


def default_device() -> str:
    return "cuda" if gpu_available() else "cpu"


def free_gpu(model: object) -> None:
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def _seconds(name: str) -> Callable[..., int]:
    """Duration for a GPU call, read from `serve.gpu_seconds` in the engine's settings."""

    def duration(engine: "Engine", *args: object, **kwargs: object) -> int:
        return int(getattr(engine.settings.thresholds.serve.gpu_seconds, name))

    return duration


def load_pinned_memory(serve: ServeThresholds) -> Memory | None:
    """The memory version pinned by the pointer file in `serve.memory_dir`; None when there is no
    pointer or the file it names is missing (the agent then runs without experience)."""
    directory = Path(serve.memory_dir)
    version = read_pointer(directory / serve.memory_pointer)
    if version is None:
        return None
    path = memory_path(directory, version)
    return load_memory(path) if path.exists() else None


def profile_from_plan(plan: ShotPlan) -> SceneProfile:
    """Rebuild the perceived profile from the key recorded in the plan's first decision."""
    scene, illum, vis, scale_, density = plan.decisions[0].split(": ", 1)[1].split("|")
    return SceneProfile.model_validate(
        {
            "scene_label": scene,
            "illumination": illum,
            "visibility": vis,
            "object_scale": scale_,
            "object_density": density,
            "confidence": 1.0,
        }
    )


def draw_detections(image: Image, detections: Sequence[Detection]) -> Image:
    """Copy of `image` with each box and its label drawn."""
    pil = PILImage.fromarray(image)
    draw = ImageDraw.Draw(pil)
    width = max(MIN_LINE_PX, max(image.shape[:2]) // LINE_DIVISOR)
    for d in detections:
        draw.rectangle(tuple(d.box), outline=BOX_COLOUR, width=width)
        text = f"{d.label} {d.score:.2f}"
        draw.text((d.box.x1 + width, d.box.y1 + width), text, fill=LABEL_COLOUR)
    return np.asarray(pil, dtype=np.uint8)


def trace_rows(events: Sequence[TraceEvent]) -> list[list[str | float | bool]]:
    return [
        [
            e.node,
            e.decision,
            e.rationale,
            round(e.duration_ms, 1),
            round(e.gpu_ms, 1),
            e.fallback_used,
        ]
        for e in events
    ]


TRACE_COLUMNS = ["node", "decision", "rationale", "duration ms", "GPU ms", "fallback"]


class Engine:
    def __init__(
        self,
        settings: Settings,
        registry: Mapping[str, RegistryEntry],
        *,
        loader: Loader | None = None,
        store: SessionStore | None = None,
        feedback_root: Path | None = None,
        memory: Memory | None = None,
    ) -> None:
        self.settings = settings
        self.registry = dict(registry)
        self._loader: Loader = loader or (lambda entry: load_adapter(entry, default_device()))
        serve = settings.thresholds.serve
        self.store = store or SessionStore(
            Path(os.environ.get(SESSIONS_ENV_VAR) or Path(tempfile.gettempdir()) / "discern"),
            serve.ttl_seconds,
        )
        self.feedback = FeedbackStore(
            feedback_root or Path(os.environ.get(FEEDBACK_ENV_VAR, "data/feedback"))
        )
        self.memory = memory
        self._manager: ModelManager | None = None
        self._sessions: dict[str, VideoSession] = {}
        self._cleans: dict[str, CleanResult] = {}
        self._media: dict[str, Path] = {}
        self._traces: dict[str, list[RequestTrace]] = {}
        self._records: list[SessionRecord] = []
        self._gpu_ms: dict[str, list[float]] = {}

    # ---- models ------------------------------------------------------------------------------

    @property
    def manager(self) -> ModelManager:
        if self._manager is None:
            profile = self.settings.profile
            self._manager = ModelManager(
                self.registry, profile.models, profile.vram_budget_gb, self._loader, free_gpu
            )
        return self._manager

    def warm_up(self) -> list[str]:
        """Load every model of the profile now (on ZeroGPU this is the CPU-side startup load).
        Returns the names that failed to load; they are logged, not raised."""
        failed: list[str] = []
        for role, name in self.settings.profile.models.items():
            try:
                self.manager.get(role, name)
            except Exception:
                logger.exception("warm-up could not load %s", name)
                failed.append(name)
        return failed

    def _roles(self, prefix: str) -> LazyRoles:
        models = self.settings.profile.models
        found = {
            role.removeprefix(prefix): (role, name)
            for role, name in models.items()
            if role.startswith(prefix)
        }
        return LazyRoles(lambda: self.manager, found)

    @property
    def restorers(self) -> LazyRoles:
        return self._roles("restorer_")

    @property
    def detectors(self) -> LazyRoles:
        """Detectors keyed by registry entry name."""
        models = self.settings.profile.models
        return LazyRoles(
            lambda: self.manager,
            {name: (role, name) for role, name in models.items() if role.startswith("detector_")},
        )

    @property
    def catalog(self) -> list[DetectorInfo]:
        return [
            DetectorInfo(
                name=name,
                capabilities=f"{self.registry[name].role.replace('_', ' ')} "
                f"({self.registry[name].model_id})",
                speed_class=self.registry[name].speed_class,
            )
            for name in self.detectors
        ]

    def _vlm(self) -> Any:
        return self.manager.get("agent_vlm")

    def _embedder(self) -> Any:
        return self.manager.get("embedder")

    def _experience(self, profile: SceneProfile, *nodes: Node) -> str:
        """Retrieved experience for `profile` rendered for the given nodes; empty without memory."""
        if self.memory is None:
            return ""
        return render_for(self.memory, profile.key, nodes, self.settings.thresholds.experience)

    def _plan_experience(self, profile: SceneProfile) -> str:
        return self._experience(profile, "restorer", "sr")

    def _available_detectors(self) -> list[str]:
        return [d.name for d in self.catalog if d.name in self.detectors]

    def _policy(self, profile: SceneProfile) -> DecisionPolicy | None:
        """Experience-gated decisions for `profile` (mode `experience.policy_mode`); None without
        memory, or without configuration stats in joint mode (the VLM decides)."""
        thresholds = self.settings.thresholds
        return build_policy(
            self.memory,
            profile,
            thresholds.experience,
            self._available_detectors(),  # catalog order: detect_image's ranking without priority
            thresholds.agent.top_k_detectors,
        )

    # ---- measurement and bookkeeping ---------------------------------------------------------

    def record_gpu_ms(self, name: str, ms: float) -> None:
        self._gpu_ms.setdefault(name, []).append(ms)

    def _gpu_total_ms(self) -> float:
        return sum(sum(v) for v in self._gpu_ms.values())

    def measured_gpu_seconds(self) -> float | None:
        """Total wall seconds of GPU-decorated calls so far; None until something was measured."""
        return self._gpu_total_ms() / 1000.0 if self._gpu_ms else None

    def measured_gpu_seconds_by_call(self) -> dict[str, float]:
        """Mean seconds per call of each GPU-decorated step that has been measured."""
        return {k: sum(v) / len(v) / 1000.0 for k, v in sorted(self._gpu_ms.items())}

    def _gpu_event(self, kind: str, before_ms: float) -> list[TraceEvent]:
        spent = self._gpu_total_ms() - before_ms
        if spent <= 0:
            return []
        return [
            TraceEvent(
                node="gpu_calls",
                started_at=datetime.now(UTC),
                duration_ms=spent,
                gpu_ms=spent,
                input_summary=kind,
                decision="wall time of GPU-decorated calls, including any ZeroGPU queue wait",
            )
        ]

    def _log_request(
        self,
        session_id: str | None,
        kind: str,
        events: list[TraceEvent],
        *,
        profile_key: str | None = None,
        grounded: bool = True,
    ) -> None:
        if session_id is not None:
            self._traces.setdefault(session_id, []).append(RequestTrace(kind, tuple(events)))
        self._records.append(
            SessionRecord(
                query_type=kind, profile_key=profile_key, grounded=grounded, events=events
            )
        )

    def trace_events(self, session_id: str | None) -> list[TraceEvent]:
        """Every event recorded for a session, oldest first."""
        if session_id is None:
            return []
        return [e for r in self._traces.get(session_id, []) for e in r.events]

    # ---- uploads and sessions ----------------------------------------------------------------

    def register_upload(self, path: str | Path) -> tuple[str, Path]:
        """Validate type and size, then copy into a new session directory. Returns the session id
        and stored path. A rejected upload leaves no session behind."""
        source = Path(path)
        max_mb = self.settings.thresholds.serve.max_upload_mb
        validate_upload(source.name, source.stat().st_size, max_mb)
        session_id = self.store.create()
        try:
            stored = self.store.save_upload(session_id, source.name, source.read_bytes(), max_mb)
        except Exception:
            self.store.delete(session_id)
            raise
        self._media[session_id] = stored
        return session_id, stored

    def cleanup(self) -> list[str]:
        """Delete expired sessions (files and in-memory state); returns the deleted ids."""
        deleted = self.store.cleanup_expired()
        for sid in deleted:
            for table in (self._sessions, self._cleans, self._media, self._traces):
                table.pop(sid, None)
        return deleted

    def delete_session(self, session_id: str) -> None:
        """Delete one session now (files and in-memory state). An unknown id raises `KeyError`."""
        self.store.delete(session_id)
        for table in (self._sessions, self._cleans, self._media, self._traces):
            table.pop(session_id, None)

    def _touch(self, session_id: str) -> None:
        try:
            self.store.touch(session_id)
        except KeyError:
            raise EngineError("this session has expired; please upload again") from None

    def session(self, session_id: str | None) -> VideoSession:
        if session_id is None or session_id not in self._sessions:
            raise EngineError("no video is loaded; upload and clean a video first")
        self._touch(session_id)
        return self._sessions[session_id]

    def clean_result(self, session_id: str | None) -> CleanResult:
        if session_id is None or session_id not in self._cleans:
            raise EngineError("no image is loaded; upload and clean an image first")
        self._touch(session_id)
        return self._cleans[session_id]

    # ---- image: clean and detect -------------------------------------------------------------

    @gpu(duration=_seconds("clean_image"))
    def _clean_gpu(self, working: Image) -> tuple[ShotPlan, Image, list[TraceEvent]]:
        trace = TraceCollector()
        plan, chosen = plan_image(
            self._vlm(),
            trace,
            working,
            self.restorers,
            experience=self._plan_experience,
            settings=self.settings,
            policy=self._policy,
        )
        cleaned = chosen
        if plan.sr_factor and "super_resolver" in self.settings.profile.models:
            with trace.span("super_resolve") as span:
                span.input_summary = f"{chosen.shape[1]}x{chosen.shape[0]}, x{plan.sr_factor}"
                cleaned = self.manager.get("super_resolver").upscale(chosen, plan.sr_factor)  # type: ignore[attr-defined]
                span.decision = f"{cleaned.shape[1]}x{cleaned.shape[0]}"
        return plan, cleaned, trace.events

    def clean_image(self, path: str | Path, session_id: str | None = None) -> CleanResult:
        """Plan SAIR for one image and apply the plan (restoration, then super-resolution when the
        plan asks for it). The agent sees a working copy no larger than the profile allows."""
        try:
            with PILImage.open(path) as pil:
                max_pixels = self.settings.thresholds.serve.max_pixels
                if pil.width * pil.height > max_pixels:  # only the header has been read
                    raise LimitError(
                        f"this image is {pil.width}x{pil.height}; at most {max_pixels} pixels "
                        "are accepted"
                    )
                original = np.asarray(pil.convert("RGB"), dtype=np.uint8)
        except OSError as err:
            raise EngineError("this file could not be read as an image") from err
        h, w = original.shape[:2]
        wanted = working_size(w, h, self.settings.profile.max_long_side_px)
        working = (
            original
            if wanted == (w, h)
            else np.asarray(PILImage.fromarray(original).resize(wanted), dtype=np.uint8)
        )
        before = self._gpu_total_ms()
        plan, cleaned, events = self._clean_gpu(working)
        events = events + self._gpu_event("clean_image", before)
        result = CleanResult(
            original, working, cleaned, plan, profile_from_plan(plan), events, session_id
        )
        if session_id is not None:
            self._cleans[session_id] = result
        self._log_request(session_id, "clean_image", events, profile_key=result.profile.key)
        return result

    @gpu(duration=_seconds("detect"))
    def _detect_gpu(
        self, cleaned: Image, targets: list[str], profile: SceneProfile
    ) -> tuple[list[Detection], list[TraceEvent]]:
        trace = TraceCollector()
        decision = detector_decision(
            self._policy(profile),
            self.settings.thresholds.agent.top_k_detectors,
            self._available_detectors(),
        )
        if decision is not None:
            record_decision(trace, decision)
        found = detect_image(
            self._vlm(),
            trace,
            cleaned,
            targets,
            profile,
            self.detectors,
            self.catalog,
            settings=self.settings,
            experience=self._experience(profile, "detector_set"),
            preferred=None if decision is None else decision.detectors,
        )
        return found, trace.events

    def detect_targets(self, result: CleanResult, targets: Sequence[str]) -> DetectResult:
        """MED on the cleaned image. Boxes are mapped back into the uploaded image's pixels."""
        wanted = [t.strip() for t in targets if t.strip()]
        if not wanted:
            raise EngineError("enter at least one target, for example: car, person")
        before = self._gpu_total_ms()
        found, events = self._detect_gpu(result.cleaned, wanted, result.profile)
        events = events + self._gpu_event("detect", before)
        detections = [d.model_copy(update={"box": result.to_original(d.box)}) for d in found]
        if result.session_id is not None:
            self._log_request(result.session_id, "detect", events)
        return DetectResult(draw_detections(result.original, detections), detections, events)

    # ---- video: ingest and ask ---------------------------------------------------------------

    @gpu(duration=_seconds("ingest"))
    def _ingest_gpu(
        self, path: Path
    ) -> tuple[VideoInfo, list[Shot], dict[int, ShotPlan], list[TraceEvent]]:
        trace = TraceCollector()
        ingest = ingest_video(
            path,
            self._vlm(),
            trace,
            self.restorers,
            experience=self._plan_experience,
            settings=self.settings,
            policy=self._policy,
        )
        return ingest.info, ingest.shots, ingest.plans, trace.events

    @gpu(duration=_seconds("index"))
    def _index_gpu(self, ingest: VideoIngest) -> tuple[VideoIndex, list[TraceEvent]]:
        trace = TraceCollector()
        return build_index(ingest, self._embedder(), self._vlm(), trace), trace.events

    def ingest(
        self, video_path: str | Path, progress: ProgressCallback | None = None
    ) -> VideoSession:
        """Validate, store, plan per shot, and index a video. Two GPU calls: planning, indexing."""
        report = progress or (lambda fraction, message: None)
        report(0.0, "checking the upload")
        session_id, stored = self.register_upload(video_path)
        try:
            return self._ingest_stored(session_id, stored, report)
        except Exception:
            self.store.delete(session_id)
            self._media.pop(session_id, None)
            raise

    def ingest_uploaded(
        self, session_id: str, progress: ProgressCallback | None = None
    ) -> VideoSession:
        """`ingest` for a video already stored by `register_upload`. A failure keeps the session."""
        report = progress or (lambda fraction, message: None)
        report(0.0, "checking the upload")
        self._touch(session_id)
        stored = self._media.get(session_id)
        if stored is None:
            raise EngineError("no video was uploaded for this session")
        return self._ingest_stored(session_id, stored, report)

    def _ingest_stored(
        self, session_id: str, stored: Path, report: ProgressCallback
    ) -> VideoSession:
        before = self._gpu_total_ms()
        report(0.1, "detecting shots and planning cleaning")
        info, shots, plans, events = self._ingest_gpu(stored)
        ingest = VideoIngest(stored, info, self.settings, shots, plans, self.restorers)
        report(0.5, "indexing frames")
        index, index_events = self._index_gpu(ingest)
        events = events + index_events + self._gpu_event("ingest", before)
        session = VideoSession(session_id, ingest, index)
        self._sessions[session_id] = session
        first = plans[shots[0].id]
        self._log_request(session_id, "ingest", events, profile_key=profile_from_plan(first).key)
        report(1.0, "ready")
        return session

    @gpu(duration=_seconds("clean_image"))
    def _previews_gpu(self, ingest: VideoIngest) -> list[tuple[Image, Image]]:
        wanted = {s.keyframe_index: s.id for s in ingest.shots}
        pairs: dict[int, tuple[Image, Image]] = {}
        profile = self.settings.profile
        for frame in iter_sampled_frames(
            ingest.path, profile.sample_fps, profile.max_long_side_px, decode_limit(self.settings)
        ):
            if frame.index not in wanted:
                continue
            plan = ingest.plans[wanted[frame.index]]
            after = frame.image
            if plan.use_restored and plan.restorer != NONE:
                try:
                    after = self.restorers[plan.restorer].restore(frame.image)
                except KeyError:
                    pass
            pairs[wanted[frame.index]] = (frame.image, after)
            if len(pairs) == len(wanted):
                break
        return [pairs[s.id] for s in ingest.shots if s.id in pairs]

    def shot_previews(self, session: VideoSession) -> list[tuple[Image, Image]]:
        """(before, after) keyframe of every shot, with that shot's plan applied to the after."""
        self._touch(session.session_id)
        before = self._gpu_total_ms()
        pairs = self._previews_gpu(session.ingest)
        self._log_request(session.session_id, "previews", self._gpu_event("previews", before))
        return pairs

    def _frame_detector(self, targets: Sequence[str]) -> Callable[[Shot, Image], list[Detection]]:
        """Fused proposals from the fast-class detectors of the profile (no VLM per frame)."""
        fast = [n for n in self.detectors if self.registry[n].speed_class == "fast"]
        floor = max(self.settings.thresholds.video.detection_threshold, SCORE_FLOOR)

        def detect(shot: Shot, image: Image) -> list[Detection]:
            pooled: list[Detection] = []
            for name in fast:
                try:
                    detector = self.detectors[name]
                except KeyError:
                    continue
                pooled += [d for d in detector.detect(image, targets) if d.score >= floor]
            groups = group_detections(image, pooled, self.settings.thresholds.grouping)
            return [g.anchor for g in groups]

        return detect

    @gpu(duration=_seconds("ask"))
    def _ask_gpu(
        self, session: VideoSession, question: str
    ) -> tuple[ResultSet | None, str, ConversationState, list[TraceEvent]]:
        trace = TraceCollector()
        vlm, embedder = self._vlm(), self._embedder()
        ingest = session.ingest

        def detect(targets: Sequence[str], segments: Sequence[Segment] | None) -> list[Track]:
            windows = None if segments is None else [(s.t_start, s.t_end) for s in segments]
            return track_video(
                ingest, self._frame_detector(targets), embedder, vlm, trace, targets, windows
            )

        services = Services(
            index=session.index,
            embedder=embedder,
            vlm=vlm,
            trace=trace,
            settings=self.settings,
            frame_size=(ingest.info.width, ingest.info.height),
            detect=detect,
        )
        state = session.conversation
        result, text = answer_question(state, question, services)
        return result, text, state, trace.events

    def ask(self, session: VideoSession, question: str) -> tuple[str, Evidence]:
        """Answer one question (follow-ups share the session's conversation state)."""
        question = question.strip()
        if not question:
            raise EngineError("type a question first")
        limit = self.settings.thresholds.serve.max_queries_per_session
        if session.queries_asked >= limit:
            raise LimitError(f"this session allows {limit} questions; upload the video again")
        self._touch(session.session_id)
        before = self._gpu_total_ms()
        result, text, state, events = self._ask_gpu(session, question)
        session.conversation = state
        events = events + self._gpu_event("ask", before)
        evidence = _evidence(result, events)
        kind = result.facts.query_type if result else "clarification"
        self._log_request(
            session.session_id, kind, events, grounded=result.grounded if result else True
        )
        return text, evidence

    def annotated_video(self, session: VideoSession, result_set_id: str) -> Path:
        """Render the result set's accepted tracks onto the uploaded video (CPU)."""
        result = session.conversation.result_sets[result_set_id]
        destination = self.store.path(session.session_id) / f"annotated-{result_set_id}.mp4"
        render_video(session.ingest.path, result.tracks, destination)
        return destination

    # ---- feedback, monitoring, About ---------------------------------------------------------

    def submit_feedback(self, session_id: str | None, feedback: Feedback) -> StoredFeedback:
        """Store feedback. The session's media is kept only when `feedback.retain_media` is set."""
        media = self._media.get(session_id) if session_id and feedback.retain_media else None
        update: dict[str, Any] = {}
        if session_id:
            update["session_ref"] = hashlib.sha256(session_id.encode()).hexdigest()[:16]
            clean = self._cleans.get(session_id)
            if feedback.image_size is None and clean is not None:
                update["image_size"] = (clean.original.shape[1], clean.original.shape[0])
        return self.feedback.add(feedback.model_copy(update=update), media)

    def monitoring_markdown(self) -> str:
        weights = self.settings.thresholds.experience.similarity_weights
        drift = self.settings.thresholds.monitor.drift_similarity
        memory = self.memory or Memory(
            version=MemoryVersion(
                id="none", created=datetime.now(UTC), record_count=0, source_hash=""
            ),
            stats=(),
        )
        text = render_markdown(summarise(self._records, memory, weights, drift))
        if self.memory is None:
            text = "\n".join(
                "Drift: not measured (no experience memory is loaded)."
                if line.startswith("Uploads without a close experience match")
                else line
                for line in text.splitlines()
            )
        return text

    def limits_markdown(self) -> str:
        p, s = self.settings.profile, self.settings.thresholds.serve
        return (
            f"Profile `{p.name}`: videos up to {p.max_video_seconds} s, frames downscaled to "
            f"{p.max_long_side_px} px on the long side, uploads up to {s.max_upload_mb:g} MB, "
            f"{s.max_queries_per_session} questions per video."
        )

    def non_commercial_models(self) -> list[RegistryEntry]:
        """Registry entries of the active profile whose license restricts commercial use."""
        used = set(self.settings.profile.models.values())
        return [
            e
            for e in self.registry.values()
            if e.name in used and any(m in e.license.lower() for m in _NON_COMMERCIAL_MARKERS)
        ]

    def about_markdown(self) -> str:
        s = self.settings.thresholds.serve
        minutes = s.ttl_seconds / _SECONDS_PER_MINUTE
        lines = [
            "## What is stored and for how long",
            "",
            f"- Uploaded media and per-session results live in a temporary session directory and "
            f"are deleted {minutes:g} minutes after the session was last used "
            f"(expired sessions are swept every {s.cleanup_interval_seconds:g} s).",
            "- Media is kept for training only if you tick the retain-media box when sending "
            "feedback. Without it, only labels and box coordinates are stored.",
            "",
            "## Limits",
            "",
            self.limits_markdown(),
            "",
            "## Licenses of non-commercial models in this profile",
            "",
        ]
        restricted = self.non_commercial_models()
        lines += [f"- {e.name} ({e.model_id}): {e.license}" for e in restricted] or [
            "- none in the active profile"
        ]
        lines += ["", "## Measured GPU time", ""]
        per_call = self.measured_gpu_seconds_by_call()
        total = self.measured_gpu_seconds()
        if total is None:
            lines.append("Not measured yet.")
        else:
            lines.append(
                "Wall time of GPU-decorated calls in this process (includes ZeroGPU queue wait):"
            )
            lines += [
                f"- {name}: mean {sec:.2f} s over {len(self._gpu_ms[name])} calls"
                for name, sec in per_call.items()
            ]
        return "\n".join(lines)


def _evidence(result: ResultSet | None, events: list[TraceEvent]) -> Evidence:
    if result is None:
        return Evidence(None, True, None, [], events)
    tracks = [
        TrackEvidence(
            t.id, t.label, t.t_start, t.t_end, t.best_crops[0].image if t.best_crops else None
        )
        for t in result.tracks
    ]
    return Evidence(result.id, result.grounded, result.facts.count, tracks, events)


def build_engine(profile_name: str | None = None) -> Engine:
    """Engine for the active profile (`DISCERN_PROFILE` when no name is given)."""
    settings = load_settings(profile_name)
    return Engine(
        settings, load_registry(), memory=load_pinned_memory(settings.thresholds.serve)
    )
