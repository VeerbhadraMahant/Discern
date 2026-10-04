import ast
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from PIL import Image as PILImage

import discern.serve.gpu as gpu_module
from discern.agent.schemas import SceneProfile
from discern.config.settings import Settings, load_settings
from discern.experience.aggregate import Memory, MemoryVersion, OptionStat, write_memory
from discern.experience.promotion import write_pointer
from discern.experience.schema import Node
from discern.feedback.schema import Feedback, feedback_to_labelled_samples
from discern.models.fakes import (
    FakeDetector,
    FakeEmbedder,
    FakeRestorer,
    FakeSuperResolver,
    FakeVLM,
)
from discern.models.manager import RegistryEntry
from discern.models.registry import load_registry
from discern.models.roles import Detection
from discern.query.schemas import ConversationState, Turn
from discern.serve.engine import (
    Engine,
    EngineError,
    LimitError,
    VideoSession,
    _seconds,
    load_pinned_memory,
)
from discern.serve.gpu import gpu
from discern.serve.session import SessionStore, UploadRejected
from discern.vision.boxes import Box
from tests.synth_video import ColourEmbedder, frame_with_box, settings_for, write_clip

SETTINGS = load_settings("local_lite")
SR_FACTOR = 4  # sr_select clamps to 4 for a small image; the fake VLM agrees
IMAGE_SIZE = (96, 64)  # width, height of the test image
DETECTOR_BOX = (40.0, 40.0, 120.0, 120.0)  # on the 4x upscaled image
BRIGHTEN = 10  # FakeRestorer default offset
BASE_LEVEL = 100


def profile_json(label: str = "fog") -> str:
    return SceneProfile.model_validate(
        {
            "scene_label": label,
            "illumination": "normal",
            "visibility": "poor",
            "object_scale": "small",
            "object_density": "sparse",
            "confidence": 0.9,
        }
    ).model_dump_json()


CLEAN_SCRIPT = [
    profile_json("fog"),
    '{"restorer": "dehaze", "rationale": "haze"}',
    '{"choice": "restored", "rationale": "clearer"}',
    f'{{"factor": {SR_FACTOR}, "rationale": "small"}}',
]
DETECT_SCRIPT = [
    '{"detectors": ["yolo-world-v2", "owlv2-base"], "rationale": "both"}',
    '{"candidate": 1, "label": "car", "reject": false, "rationale": "a car"}',
]


def make_loader(
    vlm: FakeVLM, embedder: Any = None, detections: list[Detection] | None = None
) -> Any:
    def loader(entry: RegistryEntry) -> object:
        if entry.role == "agent_vlm":
            return vlm
        if entry.role == "embedder":
            return embedder
        if entry.role.startswith("detector_"):
            return FakeDetector(entry.name, detections or [])
        if entry.role.startswith("restorer_"):
            return FakeRestorer(entry.name)
        if entry.role == "super_resolver":
            return FakeSuperResolver()
        raise AssertionError(f"unexpected role {entry.role}")

    return loader


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_engine(
    tmp_path: Path,
    vlm: FakeVLM,
    settings: Settings = SETTINGS,
    clock: Clock | None = None,
    **loader_args: Any,
) -> Engine:
    store = SessionStore(
        tmp_path / "sessions", settings.thresholds.serve.ttl_seconds, clock or Clock()
    )
    return Engine(
        settings,
        load_registry(),
        loader=make_loader(vlm, **loader_args),
        store=store,
        feedback_root=tmp_path / "feedback",
    )


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_module, "gpu_available", lambda: False)
    monkeypatch.delenv("SPACE_ID", raising=False)


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    path = tmp_path / "photo.png"
    pixels = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), BASE_LEVEL, dtype=np.uint8)
    PILImage.fromarray(pixels).save(path)
    return path


def detection(detector: str) -> Detection:
    return Detection(box=Box(*DETECTOR_BOX), label="car", score=0.9, detector=detector)


# ---- clean_image and detect_targets -----------------------------------------------------------


def test_clean_image_applies_restoration_and_super_resolution(
    tmp_path: Path, image_path: Path
) -> None:
    engine = make_engine(tmp_path, FakeVLM(CLEAN_SCRIPT))
    result = engine.clean_image(image_path)
    assert result.plan.restorer == "dehaze" and result.plan.use_restored
    assert result.plan.sr_factor == SR_FACTOR
    assert result.cleaned.shape[:2] == (IMAGE_SIZE[1] * SR_FACTOR, IMAGE_SIZE[0] * SR_FACTOR)
    assert int(result.cleaned.mean()) == BASE_LEVEL + BRIGHTEN  # the fake restorer ran
    assert result.original.shape[:2] == (IMAGE_SIZE[1], IMAGE_SIZE[0])
    assert result.profile.scene_label == "fog"
    assert {"perception", "restorer_select", "image_select", "sr_select"} <= {
        e.node for e in result.events
    }


def test_clean_image_rejects_a_file_that_is_not_an_image(tmp_path: Path) -> None:
    bad = tmp_path / "x.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(EngineError):
        make_engine(tmp_path, FakeVLM([])).clean_image(bad)


def test_image_over_the_pixel_cap_is_refused_before_decoding(
    tmp_path: Path, image_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve = SETTINGS.thresholds.serve.model_copy(
        update={"max_pixels": IMAGE_SIZE[0] * IMAGE_SIZE[1] - 1}
    )
    settings = SETTINGS.model_copy(
        update={"thresholds": SETTINGS.thresholds.model_copy(update={"serve": serve})}
    )

    def fail(self: PILImage.Image, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("pixels were decoded")

    monkeypatch.setattr(PILImage.Image, "load", fail)
    engine = make_engine(tmp_path, FakeVLM([]), settings=settings)
    with pytest.raises(LimitError, match="pixels"):
        engine.clean_image(image_path)


def test_detect_targets_scales_boxes_back_to_the_original_frame(
    tmp_path: Path, image_path: Path
) -> None:
    dets = [detection("yolo-world-v2")]
    engine = make_engine(tmp_path, FakeVLM(CLEAN_SCRIPT + DETECT_SCRIPT), detections=dets)
    result = engine.clean_image(image_path)
    out = engine.detect_targets(result, ["car", " "])
    assert len(out.detections) == 1
    assert tuple(out.detections[0].box) == tuple(v / SR_FACTOR for v in DETECTOR_BOX)
    assert out.annotated.shape == result.original.shape
    assert (out.annotated != result.original).any()  # a box was drawn


def test_detect_targets_needs_a_target(tmp_path: Path, image_path: Path) -> None:
    engine = make_engine(tmp_path, FakeVLM(CLEAN_SCRIPT))
    result = engine.clean_image(image_path)
    with pytest.raises(EngineError):
        engine.detect_targets(result, [" ", ""])


# ---- gpu decorator ----------------------------------------------------------------------------


def test_gpu_is_a_no_op_off_space() -> None:
    @gpu(duration=7)
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5 and add.__name__ == "add"


def test_gpu_is_a_no_op_on_a_space_without_the_spaces_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPACE_ID", "owner/space")
    monkeypatch.setitem(sys.modules, "spaces", None)  # makes `import spaces` raise ImportError
    assert not gpu_module.on_space()

    @gpu(duration=7)
    def one() -> int:
        return 1

    assert one() == 1


def test_gpu_wraps_with_spaces_gpu_on_a_space(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    fake = ModuleType("spaces")

    def GPU(duration: int) -> Any:  # noqa: N802  (mirrors the real name)
        def decorate(fn: Any) -> Any:
            calls.append(duration)
            return fn

        return decorate

    fake.GPU = GPU  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spaces", fake)
    monkeypatch.setenv("SPACE_ID", "owner/space")

    @gpu(duration=11)
    def two() -> int:
        return 2

    assert calls == [11] and two() == 2


def test_gpu_durations_come_from_config(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeVLM([]))
    gpu_seconds = SETTINGS.thresholds.serve.gpu_seconds
    for name in ("clean_image", "detect", "ingest", "index", "ask"):
        assert _seconds(name)(engine) == getattr(gpu_seconds, name)


# ---- measurement and About --------------------------------------------------------------------


def test_nothing_is_measured_until_a_gpu_call_runs(tmp_path: Path, image_path: Path) -> None:
    engine = make_engine(tmp_path, FakeVLM(CLEAN_SCRIPT))
    assert engine.measured_gpu_seconds() is None
    assert "Not measured yet" in engine.about_markdown()
    engine.clean_image(image_path)  # no GPU on this runtime, so still nothing measured
    assert engine.measured_gpu_seconds() is None
    assert engine.measured_gpu_seconds_by_call() == {}


def test_measured_gpu_time_is_recorded_and_shown(
    tmp_path: Path, image_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gpu_module, "gpu_available", lambda: True)
    engine = make_engine(tmp_path, FakeVLM(CLEAN_SCRIPT))
    result = engine.clean_image(image_path)
    measured = engine.measured_gpu_seconds()
    assert measured is not None and measured > 0
    assert set(engine.measured_gpu_seconds_by_call()) == {"_clean_gpu"}
    assert any(e.node == "gpu_calls" and e.gpu_ms > 0 for e in result.events)
    assert "_clean_gpu" in engine.about_markdown()


def test_about_lists_non_commercial_models_from_the_registry(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeVLM([]))
    names = {e.name for e in engine.non_commercial_models()}
    assert "zero-dce-pp" in names and "mprnet-derain" in names
    assert "swinir-denoise" not in names
    text = engine.about_markdown()
    assert "zero-dce-pp" in text and "CC-BY-NC-4.0" in text


def test_monitoring_says_drift_is_not_measured_without_a_memory(
    tmp_path: Path, image_path: Path
) -> None:
    engine = make_engine(tmp_path, FakeVLM(CLEAN_SCRIPT))
    engine.clean_image(image_path, engine.register_upload(image_path)[0])
    text = engine.monitoring_markdown()
    assert "Requests: 1" in text and "Drift: not measured" in text


# ---- uploads, limits, TTL ---------------------------------------------------------------------


def test_upload_with_a_disallowed_type_is_rejected_and_leaves_no_session(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeVLM([]))
    script = tmp_path / "run.exe"
    script.write_bytes(b"x")
    with pytest.raises(UploadRejected):
        engine.register_upload(script)
    assert list(engine.store.root.iterdir()) == []


def test_upload_over_the_size_limit_is_rejected(tmp_path: Path, image_path: Path) -> None:
    serve = SETTINGS.thresholds.serve.model_copy(update={"max_upload_mb": 0.00001})
    settings = SETTINGS.model_copy(
        update={"thresholds": SETTINGS.thresholds.model_copy(update={"serve": serve})}
    )
    engine = make_engine(tmp_path, FakeVLM([]), settings=settings)
    with pytest.raises(UploadRejected):
        engine.register_upload(image_path)
    assert list(engine.store.root.iterdir()) == []


def test_question_limit_is_enforced(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, FakeVLM([]))
    limit = SETTINGS.thresholds.serve.max_queries_per_session
    state = ConversationState(
        turns=[Turn(question="q", answer="a", result_set_id=None) for _ in range(limit)]
    )
    session = VideoSession("x", None, None, state)  # type: ignore[arg-type]
    with pytest.raises(LimitError):
        engine.ask(session, "one more?")


def test_expired_sessions_are_swept_with_their_in_memory_state(
    tmp_path: Path, image_path: Path
) -> None:
    clock = Clock()
    engine = make_engine(tmp_path, FakeVLM(CLEAN_SCRIPT), clock=clock)
    sid, stored = engine.register_upload(image_path)
    engine.clean_image(stored, sid)
    assert engine.cleanup() == []
    assert engine.clean_result(sid) is not None
    clock.now += SETTINGS.thresholds.serve.ttl_seconds + 1
    assert engine.cleanup() == [sid]
    assert not stored.exists()
    with pytest.raises(EngineError):
        engine.clean_result(sid)


def test_using_a_session_extends_its_life(tmp_path: Path, image_path: Path) -> None:
    clock = Clock()
    engine = make_engine(tmp_path, FakeVLM(CLEAN_SCRIPT), clock=clock)
    sid, stored = engine.register_upload(image_path)
    engine.clean_image(stored, sid)
    ttl = SETTINGS.thresholds.serve.ttl_seconds
    clock.now += ttl * 0.75
    engine.clean_result(sid)  # touches the session
    clock.now += ttl * 0.75
    assert engine.cleanup() == []


# ---- video: ingest, ask -----------------------------------------------------------------------


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    frames = [frame_with_box((10, 30, 34, 62)) for _ in range(20)]
    return write_clip(tmp_path / "clip.mp4", frames, fps=10)


class TextColourEmbedder(ColourEmbedder):
    def embed_text(self, texts: Any) -> Any:
        return np.ones((len(texts), 3), dtype=np.float32)


def test_ingest_ask_and_render(tmp_path: Path, clip: Path) -> None:
    vlm = FakeVLM(
        [
            profile_json("normal"),
            '{"factor": "off", "rationale": "ok"}',
            '{"caption": "a red box"}',
            '{"query_type": "count", "targets": ["car"]}',
            '{"accept": true, "label": "car", "rationale": "a car"}',
            '{"answer": "I counted 1 distinct car."}',
        ]
    )
    detections = [Detection(box=Box(10, 30, 34, 62), label="car", score=0.9, detector="d")]
    engine = make_engine(
        tmp_path, vlm, settings_for(), embedder=TextColourEmbedder(), detections=detections
    )
    seen: list[float] = []
    session = engine.ingest(clip, lambda fraction, message: seen.append(fraction))
    assert seen[0] == 0.0 and seen[-1] == 1.0
    assert len(session.ingest.shots) == 1 and session.index.captions[0] == "a red box"
    assert {e.node for e in engine.trace_events(session.session_id)} >= {"video_ingest"}

    text, evidence = engine.ask(session, "How many cars?")
    assert text == "I counted 1 distinct car."
    assert evidence.grounded and evidence.count == 1
    assert [t.label for t in evidence.tracks] == ["car"]
    assert session.queries_asked == 1
    assert engine.annotated_video(session, evidence.result_set_id or "").is_file()


def frames_scanned(
    tmp_path: Path, clip: Path, monkeypatch: pytest.MonkeyPatch, query_type: str
) -> int:
    """Frames the fast detectors were run on while answering one question about `clip`."""
    calls: list[int] = []
    original = FakeDetector.detect

    def counting(self: FakeDetector, image: Any, targets: Any) -> list[Detection]:
        calls.append(1)
        return original(self, image, targets)

    monkeypatch.setattr(FakeDetector, "detect", counting)
    vlm = FakeVLM(
        [
            profile_json("normal"),
            '{"factor": "off", "rationale": "ok"}',
            '{"caption": "a red box"}',
            f'{{"query_type": "{query_type}", "targets": ["car"]}}',
            '{"accept": true, "label": "car", "rationale": "a car"}',
            '{"answer": "There is 1 car."}',
        ]
    )
    detections = [Detection(box=Box(10, 30, 34, 62), label="car", score=0.9, detector="d")]
    base = settings_for()
    index = base.thresholds.index.model_copy(update={"top_segments": 1})
    settings = base.model_copy(
        update={"thresholds": base.thresholds.model_copy(update={"index": index})}
    )
    engine = make_engine(
        tmp_path, vlm, settings, embedder=TextColourEmbedder(), detections=detections
    )
    session = engine.ingest(clip)
    calls.clear()
    _, evidence = engine.ask(session, "question")
    assert evidence.count == 1  # found inside the segment, so no full-scan fallback ran
    return len(calls)


def test_locate_scans_only_retrieved_segments_but_count_scans_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = [frame_with_box((10, 30, 34, 62)) for _ in range(40)]  # 4 s, 20 sampled frames
    clip = write_clip(tmp_path / "long.mp4", frames, fps=10)
    full = frames_scanned(tmp_path / "count", clip, monkeypatch, "count")
    pruned = frames_scanned(tmp_path / "locate", clip, monkeypatch, "locate")
    assert 0 < pruned < full


def test_failed_ingest_leaves_no_session(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    engine = make_engine(tmp_path, FakeVLM([]), embedder=FakeEmbedder())
    with pytest.raises(ValueError):
        engine.ingest(broken)
    assert list(engine.store.root.iterdir()) == []


# ---- feedback and privacy ---------------------------------------------------------------------


def feedback(retain: bool) -> Feedback:
    return Feedback(result_set_id="image", verdict="correct", label="car", retain_media=retain)


def test_media_is_not_retained_without_opt_in(tmp_path: Path, image_path: Path) -> None:
    engine = make_engine(tmp_path, FakeVLM([]))
    sid, _ = engine.register_upload(image_path)
    stored = engine.submit_feedback(sid, feedback(False))
    assert stored.media_file is None
    assert not engine.feedback.media_dir.exists()


def test_media_is_retained_only_with_opt_in(tmp_path: Path, image_path: Path) -> None:
    engine = make_engine(tmp_path, FakeVLM([]))
    sid, _ = engine.register_upload(image_path)
    stored = engine.submit_feedback(sid, feedback(True))
    assert stored.media_file is not None
    assert (engine.feedback.media_dir / stored.media_file).is_file()


# ---- the UI module ----------------------------------------------------------------------------


def test_gradio_app_has_no_hardcoded_limits() -> None:
    """Numbers shown or enforced come from config. Only small structural literals are allowed
    outside the output-count argument of `_guard`."""
    path = Path(__file__).parents[1] / "src" / "discern" / "serve" / "gradio_app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guard_args = {
        id(n)
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_guard"
        for n in ast.walk(call)
    }
    numbers = {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, int | float)
        and not isinstance(n.value, bool)
        and id(n) not in guard_args
    }
    assert numbers <= {0, 1, 2, 3, 4}, numbers


def test_feedback_carries_a_session_reference_and_the_image_size(
    tmp_path: Path, image_path: Path
) -> None:
    """Without the size the feedback loop could never turn UI feedback into a labelled sample."""
    engine = make_engine(tmp_path, FakeVLM(CLEAN_SCRIPT))
    sid, stored_path = engine.register_upload(image_path)
    engine.clean_image(stored_path, sid)
    stored = engine.submit_feedback(
        sid,
        Feedback(
            result_set_id="image",
            verdict="correct",
            box=DETECTOR_BOX,
            label="car",
            retain_media=True,
        ),
    )
    assert stored.feedback.image_size == (IMAGE_SIZE[0], IMAGE_SIZE[1])
    assert stored.feedback.session_ref and sid not in stored.feedback.session_ref
    samples = feedback_to_labelled_samples([stored], engine.feedback)
    assert len(samples) == 1


def test_gradio_blocks_expire_their_cache_with_the_session(tmp_path: Path) -> None:
    """Gradio copies uploads into its own cache; it must be swept on the session TTL."""
    pytest.importorskip("gradio")
    from discern.serve.gradio_app import build_app

    serve = SETTINGS.thresholds.serve
    demo = build_app(make_engine(tmp_path, FakeVLM([])))
    assert demo.delete_cache == (int(serve.cleanup_interval_seconds), int(serve.ttl_seconds))


# ---- experience (DetAS-X) ---------------------------------------------------------------------


def synthetic_memory(version_id: str = "v1") -> Memory:
    key = SceneProfile.model_validate_json(profile_json("fog")).key

    def stat(node: Node, option: str, mean: float) -> OptionStat:
        return OptionStat(
            profile_key=key, query_type="detect", node=node, option=option, mean=mean, std=0.0,
            count=14,
        )

    stats = (
        stat("restorer", "dehaze", 0.52),
        stat("restorer", "none", 0.47),
        stat("sr", "4", 0.6),
        stat("sr", "off", 0.5),
        stat("detector_set", "yolo-world-v2+owlv2-base", 0.7),
    )
    version = MemoryVersion(
        id=version_id, created=datetime(2026, 1, 1, tzinfo=UTC), record_count=5, source_hash="h"
    )
    return Memory(version=version, stats=stats)


def settings_without_policy() -> Settings:
    """Settings whose policy never has enough samples, so only the experience text is exercised."""
    t = SETTINGS.thresholds
    exp = t.experience.model_copy(update={"policy_min_count": 10**6})
    return SETTINGS.model_copy(update={"thresholds": t.model_copy(update={"experience": exp})})


def test_clean_image_injects_experience_into_restorer_and_sr_prompts(
    tmp_path: Path, image_path: Path
) -> None:
    vlm = FakeVLM(CLEAN_SCRIPT)
    engine = make_engine(tmp_path, vlm, settings_without_policy())
    engine.memory = synthetic_memory()
    engine.clean_image(image_path)
    restorer_prompt, sr_prompt = vlm.prompts[1], vlm.prompts[3]
    assert "Similar scenes (1): dehaze F1 0.52 (n=14) vs none 0.47 (n=14)" in restorer_prompt
    assert "Similar scenes (1): 4 F1 0.60 (n=14) vs off 0.50 (n=14)" in sr_prompt


def test_detect_targets_injects_experience_into_the_detector_prompt(
    tmp_path: Path, image_path: Path
) -> None:
    vlm = FakeVLM(CLEAN_SCRIPT + DETECT_SCRIPT)
    engine = make_engine(
        tmp_path, vlm, settings_without_policy(), detections=[detection("yolo-world-v2")]
    )
    engine.memory = synthetic_memory()
    engine.detect_targets(engine.clean_image(image_path), ["car"])
    assert "Similar scenes (1): yolo-world-v2+owlv2-base F1 0.70 (n=14)" in vlm.prompts[4]


def test_prompts_are_unchanged_without_memory(tmp_path: Path, image_path: Path) -> None:
    with_none = FakeVLM(CLEAN_SCRIPT + DETECT_SCRIPT)
    engine = make_engine(tmp_path / "a", with_none, detections=[detection("yolo-world-v2")])
    engine.detect_targets(engine.clean_image(image_path), ["car"])
    with_memory = FakeVLM(CLEAN_SCRIPT + DETECT_SCRIPT)
    other = make_engine(
        tmp_path / "b",
        with_memory,
        settings_without_policy(),
        detections=[detection("yolo-world-v2")],
    )
    other.memory = synthetic_memory()
    other.detect_targets(other.clean_image(image_path), ["car"])
    assert with_none.prompts != with_memory.prompts
    assert not any("Similar scenes" in p for p in with_none.prompts)


def policy_memory() -> Memory:
    """Strong evidence for the fog test profile: dehaze beats none, SR off beats auto."""
    key = SceneProfile.model_validate_json(profile_json("fog")).key

    def stat(node: Node, option: str, mean: float) -> OptionStat:
        return OptionStat(
            profile_key=key, query_type="detect", node=node, option=option, mean=mean, std=0.0,
            count=50,
        )

    stats = (
        stat("restorer", "dehaze", 0.7),
        stat("restorer", "none", 0.5),
        stat("sr", "off", 0.7),
        stat("sr", "auto", 0.5),
    )
    version = MemoryVersion(
        id="v1", created=datetime(2026, 1, 1, tzinfo=UTC), record_count=4, source_hash="h"
    )
    return Memory(version=version, stats=stats)


ADJUDICATE = '{"candidate": 1, "label": "car", "reject": false, "rationale": "a car"}'


def test_strong_experience_decides_cleaning_without_those_vlm_calls(
    tmp_path: Path, image_path: Path
) -> None:
    vlm = FakeVLM([CLEAN_SCRIPT[0], ADJUDICATE])  # perception and one adjudication only
    t = SETTINGS.thresholds  # node-wise mode: the memory holds node stats only
    node_mode = t.experience.model_copy(update={"policy_mode": "node"})
    settings = SETTINGS.model_copy(
        update={"thresholds": t.model_copy(update={"experience": node_mode})}
    )
    engine = make_engine(tmp_path, vlm, settings, detections=[detection("yolo-world-v2")])
    engine.memory = policy_memory()
    result = engine.clean_image(image_path)
    assert result.plan.restorer == "dehaze" and result.plan.use_restored
    assert result.plan.sr_factor is None
    assert any(d.startswith("experience policy: sr off") for d in result.plan.decisions)
    out = engine.detect_targets(result, ["car"])
    assert len(vlm.prompts) == 2 and len(out.detections) == 1
    nodes = [e.node for e in result.events + out.events]
    assert {"experience_policy.restorer", "experience_policy.sr"} <= set(nodes)
    assert "restorer_select" not in nodes and "image_select" not in nodes


def test_without_memory_the_policy_changes_nothing(tmp_path: Path, image_path: Path) -> None:
    vlm = FakeVLM(CLEAN_SCRIPT)
    engine = make_engine(tmp_path, vlm)
    result = engine.clean_image(image_path)
    assert len(vlm.prompts) == 4 and result.plan.sr_factor == SR_FACTOR
    assert not any(e.node.startswith("experience_policy") for e in result.events)


def test_pinned_memory_is_loaded_through_the_pointer_file(tmp_path: Path) -> None:
    serve = SETTINGS.thresholds.serve.model_copy(update={"memory_dir": str(tmp_path)})
    assert load_pinned_memory(serve) is None  # no pointer
    write_pointer(tmp_path / serve.memory_pointer, "v1")
    assert load_pinned_memory(serve) is None  # pointer names a missing file
    write_memory(synthetic_memory("v1"), tmp_path)
    memory = load_pinned_memory(serve)
    assert memory is not None and memory.version.id == "v1"
