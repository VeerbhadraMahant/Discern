import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import pytest
from PIL import Image as PILImage

from discern.models.fakes import FakeVLM
from discern.models.roles import Detection
from discern.serve.api import FILE_ROUTE, GENERIC_ERROR, DiscernApi
from discern.serve.engine import Engine, EngineError
from discern.vision.boxes import Box
from tests.synth_video import frame_with_box, settings_for, write_clip
from tests.test_serve_engine import (
    CLEAN_SCRIPT,
    DETECT_SCRIPT,
    SETTINGS,
    TextColourEmbedder,
    detection,
    make_engine,
    no_gpu,  # noqa: F401  (autouse fixture)
    profile_json,
    synthetic_memory,
)

TRACE_KEYS = {
    "node",
    "started_at",
    "duration_ms",
    "gpu_ms",
    "prompt_version",
    "input_summary",
    "decision",
    "rationale",
    "fallback_used",
}
VIDEO_SCRIPT = [
    profile_json("normal"),
    '{"factor": "off", "rationale": "ok"}',
    '{"caption": "a red box"}',
    '{"query_type": "count", "targets": ["car"]}',
    '{"accept": true, "label": "car", "rationale": "a car"}',
    '{"answer": "I counted 1 distinct car."}',
]


def served_path(url: str, engine: Engine) -> Path:
    """The file a `*_url` points at, which must lie inside the session root."""
    assert url.startswith(FILE_ROUTE)
    path = Path(unquote(url.removeprefix(FILE_ROUTE)))
    assert path.resolve().is_relative_to(engine.store.root.resolve())
    assert path.is_file()
    return path


def assert_failure(out: dict[str, Any], tmp_path: Path) -> str:
    assert out["ok"] is False and set(out) == {"ok", "error"}
    text = json.dumps(out)
    assert str(tmp_path) not in text and "Traceback" not in text and ".py" not in text
    return str(out["error"])


@pytest.fixture
def image_api(tmp_path: Path) -> DiscernApi:
    vlm = FakeVLM(CLEAN_SCRIPT + DETECT_SCRIPT)
    return DiscernApi(make_engine(tmp_path, vlm, detections=[detection("yolo-world-v2")]))


@pytest.fixture
def png(tmp_path: Path) -> Path:
    path = tmp_path / "photo.png"
    PILImage.fromarray(np.full((64, 96, 3), 100, dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    frames = [frame_with_box((10, 30, 34, 62)) for _ in range(20)]
    return write_clip(tmp_path / "clip.mp4", frames, fps=10)


def video_api(tmp_path: Path) -> DiscernApi:
    detections = [Detection(box=Box(10, 30, 34, 62), label="car", score=0.9, detector="d")]
    engine = make_engine(
        tmp_path,
        FakeVLM(VIDEO_SCRIPT),
        settings_for(),
        embedder=TextColourEmbedder(),
        detections=detections,
    )
    return DiscernApi(engine)


# ---- info -------------------------------------------------------------------------------------


def test_info_reports_real_values_and_null_when_unmeasured(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeVLM([]))
    out = DiscernApi(engine).info()
    serve = SETTINGS.thresholds.serve
    assert out["ok"] is True and out["profile"] == SETTINGS.profile.name
    assert out["limits"] == {
        "max_upload_mb": serve.max_upload_mb,
        "max_video_seconds": SETTINGS.profile.max_video_seconds,
        "max_queries": serve.max_queries_per_session,
        "ttl_seconds": serve.ttl_seconds,
        "max_pixels": serve.max_pixels,
    }
    assert out["measured_gpu_seconds"] is None and out["memory_version"] is None
    licenses = {m["name"]: m["license"] for m in out["models"]}
    assert licenses["zero-dce-pp"] == "CC-BY-NC-4.0"
    assert {m["role"] for m in out["models"]} == set(SETTINGS.profile.models)
    json.dumps(out)


def test_info_names_the_pinned_memory_version(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeVLM([]))
    engine.memory = synthetic_memory("v7")
    assert DiscernApi(engine).info()["memory_version"] == "v7"


# ---- upload -----------------------------------------------------------------------------------


def test_upload_returns_a_session_and_kind(image_api: DiscernApi, png: Path) -> None:
    out = image_api.upload(str(png))
    assert out["ok"] and out["kind"] == "image" and out["name"] == "photo.png"
    assert out["size_bytes"] == png.stat().st_size and len(out["session_id"]) >= 32


def test_upload_of_a_video_is_kind_video(tmp_path: Path, clip: Path) -> None:
    assert video_api(tmp_path).upload(str(clip))["kind"] == "video"


def test_upload_rejects_a_wrong_extension(image_api: DiscernApi, tmp_path: Path) -> None:
    script = tmp_path / "run.exe"
    script.write_bytes(b"x")
    assert "not allowed" in assert_failure(image_api.upload(str(script)), tmp_path)
    assert list(image_api.engine.store.root.iterdir()) == []


def test_upload_rejects_a_file_over_the_limit(tmp_path: Path, png: Path) -> None:
    serve = SETTINGS.thresholds.serve.model_copy(update={"max_upload_mb": 0.00001})
    settings = SETTINGS.model_copy(
        update={"thresholds": SETTINGS.thresholds.model_copy(update={"serve": serve})}
    )
    api = DiscernApi(make_engine(tmp_path, FakeVLM([]), settings=settings))
    assert "exceeds" in assert_failure(api.upload(str(png)), tmp_path)


def test_upload_without_a_file_is_refused(image_api: DiscernApi, tmp_path: Path) -> None:
    assert_failure(image_api.upload(None), tmp_path)


# ---- image: clean and detect ------------------------------------------------------------------


def test_clean_then_detect_return_the_contract_shapes(image_api: DiscernApi, png: Path) -> None:
    sid = image_api.upload(str(png))["session_id"]
    cleaned = image_api.clean(sid)
    assert cleaned["ok"] is True
    served_path(cleaned["original_url"], image_api.engine)
    assert served_path(cleaned["cleaned_url"], image_api.engine).suffix == ".jpg"
    assert set(cleaned["profile"]) == {
        "scene_label",
        "illumination",
        "visibility",
        "object_scale",
        "object_density",
        "confidence",
    }
    assert cleaned["plan"]["restorer"] == "dehaze" and cleaned["plan"]["sr_factor"] == 4
    assert cleaned["plan"]["use_restored"] is True and cleaned["plan"]["decisions"]
    assert cleaned["events"] and set(cleaned["events"][0]) == TRACE_KEYS

    detected = image_api.detect(sid, "car, person")
    assert detected["ok"] is True
    served_path(detected["annotated_url"], image_api.engine)
    (found,) = detected["detections"]
    assert set(found) == {"label", "score", "box", "detector"} and len(found["box"]) == 4
    assert set(detected["events"][0]) == TRACE_KEYS
    json.dumps([cleaned, detected])


def test_media_names_are_unguessable_and_inside_the_session_directory(
    image_api: DiscernApi, png: Path
) -> None:
    sid = image_api.upload(str(png))["session_id"]
    first = served_path(image_api.clean(sid)["cleaned_url"], image_api.engine)
    assert first.parent == image_api.engine.store.path(sid)
    assert len(first.stem.rsplit("-", 1)[-1]) == 16


def test_detect_needs_a_target_and_a_cleaned_image(
    image_api: DiscernApi, png: Path, tmp_path: Path
) -> None:
    sid = image_api.upload(str(png))["session_id"]
    assert "clean" in assert_failure(image_api.detect(sid, "car"), tmp_path)
    image_api.clean(sid)
    assert "target" in assert_failure(image_api.detect(sid, " , "), tmp_path)


def test_unknown_and_malformed_session_ids_are_errors(
    image_api: DiscernApi, tmp_path: Path
) -> None:
    for sid in ("nope", "A" * 43, ""):
        for out in (
            image_api.clean(sid),
            image_api.detect(sid, "car"),
            list(image_api.ingest(sid))[-1],
            image_api.ask(sid, "q"),
            image_api.annotated_video(sid, "R1"),
            image_api.trace(sid),
            image_api.feedback(sid, "image", None, "correct", "", False),
        ):
            assert "expired" in assert_failure(out, tmp_path)


def test_a_video_session_cannot_be_cleaned_as_an_image(tmp_path: Path, clip: Path) -> None:
    api = video_api(tmp_path)
    sid = api.upload(str(clip))["session_id"]
    assert "video" in assert_failure(api.clean(sid), tmp_path)


# ---- error handling ---------------------------------------------------------------------------


def test_unexpected_errors_become_a_generic_message(
    image_api: DiscernApi, png: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = image_api.upload(str(png))["session_id"]

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"boom in {tmp_path}/secret/engine.py line 3")

    monkeypatch.setattr(image_api.engine, "clean_image", boom)
    assert assert_failure(image_api.clean(sid), tmp_path) == GENERIC_ERROR


def test_paths_in_engine_messages_are_redacted(
    image_api: DiscernApi, png: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = image_api.upload(str(png))["session_id"]

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise EngineError(f"cannot read {tmp_path}/sessions/x/upload.png or C:\\data\\x.png")

    monkeypatch.setattr(image_api.engine, "clean_image", refuse)
    message = assert_failure(image_api.clean(sid), tmp_path)
    assert message.startswith("cannot read <path>") and "x.png" not in message


# ---- video: ingest, ask, annotated video, trace -----------------------------------------------


def test_ingest_yields_increasing_progress_then_done(tmp_path: Path, clip: Path) -> None:
    api = video_api(tmp_path)
    sid = api.upload(str(clip))["session_id"]
    updates = list(api.ingest(sid))
    *progress, done = updates
    values = [u["progress"] for u in progress]
    assert len(values) >= 3 and values == sorted(set(values)) and values[0] >= 0.0
    assert values[-1] < 1.0
    assert all(u["ok"] and u["stage"] == "progress" and u["message"] for u in progress)
    assert done["ok"] is True and done["stage"] == "done"
    assert done["width"] > 0 and done["height"] > 0 and done["duration"] > 0 and done["fps"] > 0
    (shot,) = done["shots"]
    assert set(shot) == {"id", "t_start", "t_end", "profile", "plan", "before_url", "after_url"}
    served_path(shot["before_url"], api.engine)
    served_path(shot["after_url"], api.engine)
    served_path(done["video_url"], api.engine)
    assert {e["node"] for e in done["events"]} >= {"video_ingest"}
    json.dumps(updates)


def test_a_failed_ingest_ends_with_one_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    api = video_api(tmp_path)
    sid = api.upload(str(broken))["session_id"]
    *_, last = list(api.ingest(sid))
    assert_failure(last, tmp_path)


def test_ask_annotated_video_trace_and_cleanup(tmp_path: Path, clip: Path) -> None:
    api = video_api(tmp_path)
    sid = api.upload(str(clip))["session_id"]
    list(api.ingest(sid))
    out = api.ask(sid, "How many cars?")
    assert out["ok"] and out["answer"] == "I counted 1 distinct car."
    assert out["grounded"] is True and out["clarification"] is False
    assert out["result_set_id"] == "R1" and out["evidence"]["summary"]
    (track,) = out["evidence"]["tracks"]
    assert set(track) == {
        "id",
        "label",
        "t_start",
        "t_end",
        "n_frames",
        "mean_score",
        "status",
        "rationale",
        "crop_url",
    }
    assert track["label"] == "car" and track["status"] == "accepted" and track["n_frames"] > 0
    served_path(track["crop_url"], api.engine)
    assert set(out["events"][0]) == TRACE_KEYS
    json.dumps(out)

    video = api.annotated_video(sid, "R1")
    assert video["ok"] and served_path(video["video_url"], api.engine).suffix == ".mp4"
    assert api.annotated_video(sid, "R1")["video_url"] == video["video_url"]
    assert "question" in assert_failure(api.annotated_video(sid, "R9"), tmp_path)

    trace = api.trace(sid)
    assert trace["ok"] and {e["node"] for e in trace["events"]} >= {"video_ingest"}

    directory = api.engine.store.path(sid)
    assert api.cleanup(sid) == {"ok": True}
    assert not directory.exists()
    assert api.cleanup(sid) == {"ok": True}  # idempotent
    assert "expired" in assert_failure(api.trace(sid), tmp_path)


def test_ask_before_ingest_and_an_empty_question_are_errors(tmp_path: Path, clip: Path) -> None:
    api = video_api(tmp_path)
    sid = api.upload(str(clip))["session_id"]
    assert "video" in assert_failure(api.ask(sid, "q"), tmp_path)
    list(api.ingest(sid))
    assert "question" in assert_failure(api.ask(sid, "  "), tmp_path)


# ---- feedback ---------------------------------------------------------------------------------


def test_feedback_without_opt_in_stores_no_media(image_api: DiscernApi, png: Path) -> None:
    sid = image_api.upload(str(png))["session_id"]
    assert image_api.feedback(sid, "image", None, "correct", "fine", False) == {"ok": True}
    (stored,) = image_api.engine.feedback.load()
    assert stored.media_file is None and stored.feedback.note == "fine"
    assert not image_api.engine.feedback.media_dir.exists()


def test_feedback_with_opt_in_keeps_the_media(image_api: DiscernApi, png: Path) -> None:
    sid = image_api.upload(str(png))["session_id"]
    assert image_api.feedback(sid, "image", 3, "wrong", "", True)["ok"]
    (stored,) = image_api.engine.feedback.load()
    assert stored.feedback.track_id == 3 and stored.media_file is not None
    assert (image_api.engine.feedback.media_dir / stored.media_file).is_file()


def test_feedback_validation(image_api: DiscernApi, png: Path, tmp_path: Path) -> None:
    sid = image_api.upload(str(png))["session_id"]
    missing = image_api.feedback(sid, "image", None, "missing", "", False)
    assert "box" in assert_failure(missing, tmp_path)
    assert_failure(image_api.feedback(sid, "image", None, "maybe", "", False), tmp_path)
    boxed = image_api.feedback(
        sid, "image", None, "missing", "", False, box=[1, 2, 30, 40], label="car"
    )
    assert boxed["ok"]
    assert image_api.engine.feedback.load()[-1].feedback.box == (1, 2, 30, 40)


# ---- gradio wiring ----------------------------------------------------------------------------


def test_build_app_exposes_the_ten_api_names(tmp_path: Path) -> None:
    pytest.importorskip("gradio")
    from discern.serve.gradio_app import API_NAMES, build_app, launch_kwargs

    engine = make_engine(tmp_path, FakeVLM([]))
    app = build_app(engine)
    exposed = {d["api_name"]: d for d in app.config["dependencies"] if d.get("api_name")}
    assert set(API_NAMES) <= set(exposed) and len(API_NAMES) == 10
    assert exposed["discern_ingest"]["types"]["generator"] is True
    assert all(f"/{n}" in app.get_api_info()["named_endpoints"] for n in API_NAMES)
    kwargs = launch_kwargs(engine)
    assert kwargs["allowed_paths"] == [str(engine.store.root.resolve())]
    assert kwargs["strict_cors"] is True
