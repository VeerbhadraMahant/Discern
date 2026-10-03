from pathlib import Path

import numpy as np
import pytest

from discern.agent.llm_io import load_prompt
from discern.agent.nodes.caption import caption
from discern.config.settings import Settings, load_settings
from discern.index.video_index import VideoIndex, build_index, retrieve
from discern.models.fakes import FakeRestorer, FakeVLM
from discern.trace import TraceCollector
from discern.video.pipeline import ingest_video
from tests.query_fakes import ScriptedEmbedder, make_index
from tests.synth_video import (
    ColourEmbedder,
    background,
    frame_with_box,
    settings_for,
    write_clip,
)
from tests.test_video_pipeline import SR_OFF, profile_json

SNAPSHOTS = Path(__file__).parent / "snapshots"
SETTINGS = load_settings("local_lite")
WINDOW = SETTINGS.thresholds.index.smoothing_seconds


def index_with(vectors: list[tuple[float, float]]) -> VideoIndex:
    base = make_index(len(vectors))
    return VideoIndex(base.times, np.array(vectors, np.float32), base.shots, base.captions)


def settings_with_window(window: float) -> Settings:
    t = SETTINGS.thresholds
    return SETTINGS.model_copy(
        update={"thresholds": t.model_copy(update={"index": t.index.model_copy(
            update={"smoothing_seconds": window})})}
    )


def test_caption_prompt_matches_golden_snapshot() -> None:
    assert load_prompt("caption", 1).render() == (SNAPSHOTS / "caption.v1.txt").read_text(
        encoding="utf-8"
    )


def test_smoothing_prefers_a_sustained_region_over_a_single_spike() -> None:
    vectors = [(0.0, 1.0)] * 20
    for t in (4, 5, 6):
        vectors[t] = (0.8, 0.6)  # cosine 0.8 with the query
    vectors[15] = (1.0, 0.0)  # a lone frame that matches perfectly
    index = index_with(vectors)
    embedder = ScriptedEmbedder({"car": (1.0, 0.0)})

    (best,) = retrieve(index, embedder, "car", 1, settings_with_window(2.0))

    assert best.t_start == pytest.approx(4.0) and best.t_end == pytest.approx(6.0)
    assert best.score == pytest.approx(0.8)  # raw best frame would be 1.0 at t=15


def test_segments_are_ranked_best_first_and_do_not_overlap() -> None:
    vectors = [(0.0, 1.0)] * 20
    for t in (3, 4, 5):
        vectors[t] = (0.6, 0.8)
    for t in (12, 13, 14):
        vectors[t] = (0.9, 0.436)
    segments = retrieve(
        index_with(vectors), ScriptedEmbedder({"x": (1.0, 0.0)}), "x", 2, settings_with_window(2.0)
    )
    assert [round((s.t_start + s.t_end) / 2) for s in segments] == [13, 4]
    assert segments[0].score > segments[1].score
    assert segments[0].t_start >= segments[1].t_end


def test_segments_are_clipped_to_the_video() -> None:
    vectors = [(1.0, 0.0)] + [(0.0, 1.0)] * 9
    (seg,) = retrieve(
        index_with(vectors), ScriptedEmbedder({"x": (1.0, 0.0)}), "x", 1, settings_with_window(4.0)
    )
    assert seg.t_start == 0.0


def test_retrieve_on_an_empty_index_is_empty() -> None:
    empty = VideoIndex(np.zeros(0), np.zeros((0, 0), np.float32), [], {})
    assert retrieve(empty, ScriptedEmbedder({}), "x", 3) == []


def test_default_window_comes_from_config() -> None:
    vectors = [(0.0, 1.0)] * 20
    vectors[10] = (1.0, 0.0)
    (seg,) = retrieve(index_with(vectors), ScriptedEmbedder({"x": (1.0, 0.0)}), "x", 1)
    assert seg.t_end - seg.t_start == pytest.approx(WINDOW)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    index = make_index(10, {0: "a road", 1: ""})
    index.embeddings[3] = (0.6, 0.8)
    directory = tmp_path / "session" / "index"
    index.save(directory)
    loaded = VideoIndex.load(directory)
    np.testing.assert_array_equal(loaded.times, index.times)
    np.testing.assert_array_equal(loaded.embeddings, index.embeddings)
    assert loaded.shots == index.shots
    assert loaded.captions == {0: "a road", 1: ""}


def test_caption_node_and_fallback() -> None:
    img = background()
    assert caption(FakeVLM(['{"caption": " A car on a road. "}']), TraceCollector(), img) == (
        "A car on a road."
    )
    trace = TraceCollector()
    assert caption(FakeVLM(["nope", "still nope"]), trace, img) == ""
    assert trace.events[-1].fallback_used


def test_build_index_embeds_every_sampled_frame_and_captions_each_shot(tmp_path: Path) -> None:
    frames = [frame_with_box((10 + 2 * i, 30, 34 + 2 * i, 62), bg=background(1)) for i in range(20)]
    clip = write_clip(tmp_path / "clip.mp4", frames, fps=10)
    vlm = FakeVLM([profile_json("normal"), SR_OFF, '{"caption": "a red box moves right"}'])
    trace = TraceCollector()
    ingest = ingest_video(clip, vlm, trace, {"x": FakeRestorer("x")}, settings=settings_for())
    index = build_index(ingest, ColourEmbedder(), vlm, trace)
    assert len(index.times) == 10  # 2 s at 5 fps
    assert index.embeddings.shape == (10, 3)
    assert np.allclose(np.linalg.norm(index.embeddings, axis=1), 1.0)
    assert index.captions == {0: "a red box moves right"}
    assert any(e.node == "index_build" for e in trace.events)
