"""The JSON API behind the React frontend (contract: frontend/API.md). No gradio import.

`DiscernApi` has one method per endpoint. Each returns a JSON-serialisable dict and never raises:
any failure becomes `{"ok": false, "error": "<message>"}`. Messages of the engine's own refusals
(`EngineError`, `UploadRejected`, `VideoError`) are shown with any file path replaced; every other
exception is logged server-side and replaced by a generic message.

How media reaches the browser. Every image or video a response mentions is a file inside the
session directory (`SessionStore`), under an unguessable name (`<prefix>-<16 hex>.<ext>`, inside a
directory named by the 43-character session token). The `*_url` fields are root-relative Gradio
file URLs, `/gradio_api/file=<absolute posix path>`. The browser resolves them against the Gradio
root (`client.config.root`, for example the Space URL). Gradio serves such a path only if it lies
under a directory passed to `launch(allowed_paths=[...])`; `gradio_app.launch_kwargs` passes the
session root and nothing else, so no other part of the disk is exposed. Files vanish with the
session (TTL sweep or `discern_cleanup`). `file_url` is the single place that builds a URL; the
`url_for` constructor argument replaces it.
"""

import contextvars
import logging
import queue
import re
import secrets
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from PIL import Image as PILImage
from pydantic import ValidationError

from discern.feedback.schema import Feedback
from discern.serve.engine import Engine, EngineError, Evidence, VideoSession, profile_from_plan
from discern.serve.session import UploadRejected
from discern.trace import TraceEvent
from discern.video.types import Track, VideoError

logger = logging.getLogger(__name__)

FILE_ROUTE = "/gradio_api/file="
IMAGE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "webp"})
JPEG_QUALITY = 90
GENERIC_ERROR = "Something went wrong while processing this request. Please try again."
EXPIRED_ERROR = "this session is unknown or has expired; please upload again"
_SAFE_ERRORS = (EngineError, UploadRejected, VideoError)
_PATH_PATTERN = re.compile(r"(?:[A-Za-z]:)?[\\/][^\s'\"]+[\\/][^\s'\"]*")
_PROGRESS_SHARE = 0.9  # engine progress is scaled into 0..this; previews take the rest
_PREVIEW_PROGRESS = 0.95
_FEEDBACK_VERDICTS = ("correct", "wrong", "missing")
_PROFILE_FIELDS = (
    "scene_label",
    "illumination",
    "visibility",
    "object_scale",
    "object_density",
    "confidence",
)

Json = dict[str, Any]
UrlFor = Callable[[Path], str]


def file_url(path: Path) -> str:
    """Root-relative Gradio file URL of an absolute path (percent-encoded)."""
    return FILE_ROUTE + quote(path.resolve().as_posix(), safe="/:")


def _fail(message: str) -> Json:
    return {"ok": False, "error": message}


def _event(e: TraceEvent) -> Json:
    return {
        "node": e.node,
        "started_at": e.started_at.isoformat(),
        "duration_ms": e.duration_ms,
        "gpu_ms": e.gpu_ms,
        "prompt_version": e.prompt_version,
        "input_summary": e.input_summary,
        "decision": e.decision,
        "rationale": e.rationale,
        "fallback_used": e.fallback_used,
    }


def _events(events: Sequence[TraceEvent]) -> list[Json]:
    return [_event(e) for e in events]


class DiscernApi:
    def __init__(self, engine: Engine, url_for: UrlFor = file_url) -> None:
        self.engine = engine
        self._url_for = url_for
        self._uploads: dict[str, tuple[str, Path]] = {}  # session id -> (kind, stored path)
        self._annotated: dict[tuple[str, str], Path] = {}

    # ---- error handling ----------------------------------------------------------------------

    def _message(self, err: Exception) -> str:
        if isinstance(err, KeyError):  # an unknown or malformed session id
            return EXPIRED_ERROR
        if isinstance(err, _SAFE_ERRORS):
            root = self.engine.store.root
            text = str(err).replace(str(root.parent), "<path>").replace(str(root), "<path>")
            return _PATH_PATTERN.sub("<path>", text)
        logger.error("request failed", exc_info=err)
        return GENERIC_ERROR

    def _safe(self, call: Callable[[], Json]) -> Json:
        try:
            return call()
        except Exception as err:
            if isinstance(err, (*_SAFE_ERRORS, KeyError)):
                logger.info("request refused: %s", err)
            return _fail(self._message(err))

    # ---- media -------------------------------------------------------------------------------

    def _write_image(self, session_id: str, prefix: str, image: np.ndarray) -> str:
        directory = self.engine.store.path(session_id)
        path = directory / f"{prefix}-{secrets.token_hex(8)}.jpg"
        PILImage.fromarray(image).convert("RGB").save(path, quality=JPEG_QUALITY)
        return self._url_for(path)

    def _optional_image(self, session_id: str, prefix: str, image: np.ndarray | None) -> str | None:
        return None if image is None else self._write_image(session_id, prefix, image)

    def _upload(self, session_id: str) -> tuple[str, Path]:
        self.engine.store.touch(session_id)  # KeyError for unknown, malformed or expired ids
        if session_id not in self._uploads:
            raise EngineError("nothing was uploaded for this session")
        return self._uploads[session_id]

    # ---- endpoints ---------------------------------------------------------------------------

    def info(self) -> Json:
        def run() -> Json:
            e = self.engine
            profile, serve = e.settings.profile, e.settings.thresholds.serve
            return {
                "ok": True,
                "profile": profile.name,
                "limits": {
                    "max_upload_mb": serve.max_upload_mb,
                    "max_video_seconds": profile.max_video_seconds,
                    "max_queries": serve.max_queries_per_session,
                    "ttl_seconds": serve.ttl_seconds,
                    "max_pixels": serve.max_pixels,
                },
                "models": [
                    {"role": role, "name": name, "license": e.registry[name].license}
                    for role, name in profile.models.items()
                ],
                "measured_gpu_seconds": e.measured_gpu_seconds(),
                "memory_version": None if e.memory is None else e.memory.version.id,
            }

        return self._safe(run)

    def upload(self, file: str | None) -> Json:
        def run() -> Json:
            if not file:
                raise EngineError("choose an image or a video first")
            source = Path(file)
            session_id, stored = self.engine.register_upload(source)
            kind = "image" if stored.suffix.lstrip(".").lower() in IMAGE_EXTENSIONS else "video"
            self._uploads[session_id] = (kind, stored)
            return {
                "ok": True,
                "session_id": session_id,
                "kind": kind,
                "name": source.name,
                "size_bytes": stored.stat().st_size,
            }

        return self._safe(run)

    def clean(self, session_id: str) -> Json:
        def run() -> Json:
            kind, stored = self._upload(session_id)
            if kind != "image":
                raise EngineError("this upload is a video; ingest it instead")
            result = self.engine.clean_image(stored, session_id)
            profile = result.profile.model_dump(mode="json")
            plan = result.plan
            return {
                "ok": True,
                "original_url": self._url_for(stored),
                "cleaned_url": self._write_image(session_id, "cleaned", result.cleaned),
                "profile": {k: profile[k] for k in _PROFILE_FIELDS},
                "plan": {
                    "restorer": plan.restorer,
                    "use_restored": plan.use_restored,
                    "sr_factor": plan.sr_factor,
                    "decisions": list(plan.decisions),
                },
                "events": _events(result.events),
            }

        return self._safe(run)

    def detect(self, session_id: str, targets: str) -> Json:
        def run() -> Json:
            self._upload(session_id)
            result = self.engine.clean_result(session_id)
            out = self.engine.detect_targets(result, (targets or "").split(","))
            return {
                "ok": True,
                "annotated_url": self._write_image(session_id, "annotated", out.annotated),
                "detections": [
                    {
                        "label": d.label,
                        "score": float(d.score),
                        "box": [float(v) for v in d.box],
                        "detector": d.detector,
                    }
                    for d in out.detections
                ],
                "events": _events(out.events),
            }

        return self._safe(run)

    def ingest(self, session_id: str) -> Iterator[Json]:
        """Yield progress dicts (increasing `progress`), then the final `stage: done` dict. The
        engine runs in a worker thread that carries this thread's context variables (ZeroGPU
        reads the caller's identity from them); errors arrive as one `ok: false` dict."""
        updates: queue.Queue[tuple[str, Any]] = queue.Queue()
        try:
            kind, _ = self._upload(session_id)
            if kind != "video":
                raise EngineError("this upload is an image; clean it instead")
        except Exception as err:
            yield _fail(self._message(err))
            return

        def report(fraction: float, message: str) -> None:
            updates.put(("progress", (fraction * _PROGRESS_SHARE, message)))

        def work() -> None:
            try:
                updates.put(("session", self.engine.ingest_uploaded(session_id, report)))
            except Exception as err:
                updates.put(("error", err))

        context = contextvars.copy_context()
        thread = threading.Thread(target=context.run, args=(work,), daemon=True)
        thread.start()
        last = -1.0
        while True:
            tag, payload = updates.get()
            if tag == "progress":
                fraction, message = payload
                if fraction > last:
                    last = fraction
                    yield {
                        "ok": True,
                        "stage": "progress",
                        "progress": fraction,
                        "message": message,
                    }
            elif tag == "error":
                yield _fail(self._message(payload))
                return
            else:
                break
        thread.join()
        session: VideoSession = payload
        yield {
            "ok": True,
            "stage": "progress",
            "progress": _PREVIEW_PROGRESS,
            "message": "rendering shot previews",
        }
        yield self._safe(lambda: self._ingest_done(session))

    def _ingest_done(self, session: VideoSession) -> Json:
        sid, ingest = session.session_id, session.ingest
        pairs = self.engine.shot_previews(session)
        shots = []
        for i, shot in enumerate(ingest.shots):
            plan = ingest.plans[shot.id]
            before, after = pairs[i] if i < len(pairs) else (None, None)
            shots.append(
                {
                    "id": shot.id,
                    "t_start": shot.t_start,
                    "t_end": shot.t_end,
                    "profile": profile_from_plan(plan).model_dump(mode="json"),
                    "plan": {
                        "restorer": plan.restorer,
                        "use_restored": plan.use_restored,
                        "sr_factor": plan.sr_factor,
                        "decisions": list(plan.decisions),
                    },
                    "before_url": self._optional_image(sid, "before", before),
                    "after_url": self._optional_image(sid, "after", after),
                }
            )
        info = ingest.info
        return {
            "ok": True,
            "stage": "done",
            "duration": info.duration,
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
            "shots": shots,
            "video_url": self._url_for(ingest.path),
            "events": _events(self.engine.trace_events(sid)),
        }

    def ask(self, session_id: str, question: str) -> Json:
        def run() -> Json:
            self._upload(session_id)
            session = self.engine.session(session_id)
            answer, evidence = self.engine.ask(session, question or "")
            result = (
                None
                if evidence.result_set_id is None
                else session.conversation.result_sets[evidence.result_set_id]
            )
            crops = {t.track_id: t.crop for t in evidence.tracks}
            tracks = (
                [self._track(session_id, t, crops.get(t.id)) for t in result.tracks]
                if result
                else []
            )
            return {
                "ok": True,
                "answer": answer,
                "grounded": evidence.grounded,
                "clarification": result is None,
                "result_set_id": evidence.result_set_id,
                "evidence": {"tracks": tracks, "summary": self._summary(evidence, tracks)},
                "events": _events(evidence.events),
            }

        return self._safe(run)

    def _track(self, session_id: str, track: Track, crop: np.ndarray | None) -> Json:
        return {
            "id": track.id,
            "label": track.label,
            "t_start": track.t_start,
            "t_end": track.t_end,
            "n_frames": len(track.frames),
            "mean_score": track.mean_score,
            "status": track.status,
            "rationale": track.rationale,
            "crop_url": None if crop is None else self._write_image(session_id, "crop", crop),
        }

    @staticmethod
    def _summary(evidence: Evidence, tracks: list[Json]) -> str:
        if evidence.result_set_id is None:
            return "No evidence: the question needed clarification."
        parts = [f"{len(tracks)} track(s) in the evidence"]
        if evidence.count is not None:
            parts.append(f"count {evidence.count}")
        if not evidence.grounded:
            parts.append("ungrounded description: not backed by detections or tracks")
        return "; ".join(parts) + "."

    def annotated_video(self, session_id: str, result_set_id: str) -> Json:
        def run() -> Json:
            self._upload(session_id)
            session = self.engine.session(session_id)
            if result_set_id not in session.conversation.result_sets:
                raise EngineError("ask a question first")
            key = (session_id, result_set_id)
            path = self._annotated.get(key)
            if path is None or not path.is_file():
                rendered = self.engine.annotated_video(session, result_set_id)
                path = rendered.with_name(
                    f"annotated-{result_set_id}-{secrets.token_hex(8)}{rendered.suffix}"
                )
                rendered.replace(path)
                self._annotated[key] = path
            return {"ok": True, "video_url": self._url_for(path)}

        return self._safe(run)

    def trace(self, session_id: str) -> Json:
        def run() -> Json:
            self.engine.store.touch(session_id)
            return {"ok": True, "events": _events(self.engine.trace_events(session_id))}

        return self._safe(run)

    def feedback(
        self,
        session_id: str,
        result_set_id: str,
        track_id: int | None,
        verdict: str,
        note: str,
        retain_media: bool,
        box: Sequence[float] | None = None,
        label: str = "",
    ) -> Json:
        """`box` and `label` are optional additions to the contract: a "missing" verdict needs the
        visitor's box (the feedback schema requires it) and feedback on a track names its label."""

        def run() -> Json:
            self.engine.store.touch(session_id)
            if verdict not in _FEEDBACK_VERDICTS:
                raise EngineError("the verdict must be correct, wrong or missing")
            if box is not None and len(box) != 4:
                raise EngineError("the box needs four numbers: x1, y1, x2, y2")
            try:
                item = Feedback(
                    result_set_id=(result_set_id or "").strip(),
                    verdict=verdict,
                    track_id=None if track_id is None else int(track_id),
                    box=None if box is None else (box[0], box[1], box[2], box[3]),
                    label=(label or "").strip(),
                    note=(note or "").strip(),
                    retain_media=bool(retain_media),
                )
            except ValidationError as err:
                raise EngineError(err.errors()[0]["msg"].removeprefix("Value error, ")) from None
            self.engine.submit_feedback(session_id, item)
            return {"ok": True}

        return self._safe(run)

    def cleanup(self, session_id: str) -> Json:
        def run() -> Json:
            try:
                self.engine.delete_session(session_id)
            except KeyError:
                pass  # already gone: the visitor's goal is met
            self._uploads.pop(session_id, None)
            for key in [k for k in self._annotated if k[0] == session_id]:
                del self._annotated[key]
            return {"ok": True}

        return self._safe(run)
