from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from discern.agent.nodes.restorer_select import restorer_select
from discern.agent.schemas import SceneProfile
from discern.config.settings import SimilarityWeights, load_settings
from discern.eval.types import GroundTruthBox
from discern.experience.aggregate import (
    Memory,
    aggregate,
    build_memory,
    load_memory,
    memory_path,
    write_memory,
)
from discern.experience.harvest import (
    CachedOutputs,
    Configuration,
    enumerate_configurations,
    harvest_image,
)
from discern.experience.injection import render_all, render_node
from discern.experience.retrieval import profile_similarity, retrieve
from discern.experience.schema import NODE_METRIC, ExperienceRecord, ExperienceStore
from discern.models.fakes import FakeVLM
from discern.models.roles import Detection, Image
from discern.trace import TraceCollector
from discern.vision.boxes import Box

SETTINGS = load_settings()
EXP = SETTINGS.thresholds.experience
GROUPING = SETTINGS.thresholds.grouping
GT_BOX = Box(40, 30, 80, 70)


def profile(label: str = "fog", **over: str) -> SceneProfile:
    base = {
        "scene_label": label,
        "illumination": "normal",
        "visibility": "poor",
        "object_scale": "small",
        "object_density": "dense",
        "confidence": 0.8,
    }
    return SceneProfile.model_validate({**base, **over})


def det(box: Box, score: float, detector: str) -> Detection:
    return Detection(box=box, label="thing", score=score, detector=detector)


def blob_image() -> Image:
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img[30:70, 40:80] = (230, 40, 40)
    return img


def cached_outputs() -> CachedOutputs:
    """Detector a finds the object only after dehaze without SR; detector b is always wrong."""
    wrong = det(Box(200, 10, 240, 50), 0.8, "b")
    out: dict[tuple[str, str], dict[str, list[Detection]]] = {}
    for restorer in ("none", "dehaze"):
        for sr in ("off", "auto"):
            good = restorer == "dehaze" and sr == "off"
            out[(restorer, sr)] = {"a": [det(GT_BOX, 0.9, "a")] if good else [], "b": [wrong]}
    return out


def run_harvest() -> tuple[list[ExperienceRecord], list[Configuration]]:
    calls: list[Configuration] = []

    def fake_confirm(c: Configuration) -> float:
        calls.append(c)
        return 0.9 if c.variant == ("dehaze", "off") and c.detectors == ("a",) else 0.1

    records = harvest_image(
        "img1",
        profile(),
        "detect",
        blob_image(),
        [GroundTruthBox(box=GT_BOX, label="thing")],
        cached_outputs(),
        "dehaze",
        ["a", "b"],
        GROUPING,
        fake_confirm,
        EXP.confirm_top_configs,
        "v1",
    )
    return records, calls


# ---- schema and store -------------------------------------------------------------------------


def test_store_appends_and_loads(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path / "sub" / "records.jsonl")
    assert store.load() == []
    records, _ = run_harvest()
    assert store.append(records[:3]) == 3
    assert store.append(records[3:5]) == 2
    assert store.load() == records[:5]


# ---- harvest ----------------------------------------------------------------------------------


def test_enumerates_all_configurations() -> None:
    configs = enumerate_configurations("dehaze", ["b", "a", "c"])
    assert len(configs) == 2 * 2 * (3 + 3)  # restorer x sr x (singles + pairs)
    assert {c.restorer for c in configs} == {"none", "dehaze"}
    assert {c.sr for c in configs} == {"off", "auto"}
    assert ("a", "c") in {c.detectors for c in configs}
    assert len(enumerate_configurations("none", ["a", "b"])) == 2 * 3


def test_harvest_confirms_top_three_only() -> None:
    _, calls = run_harvest()
    assert len(calls) == 3
    assert calls[0].detectors == ("a",) or calls[1].detectors == ("a",)
    assert all(c.restorer == "dehaze" and c.sr == "off" for c in calls[:2])


def test_node_values_are_best_f1_given_option() -> None:
    records, _ = run_harvest()
    node = {(r.node, r.option): r.metric_value for r in records if r.metric_name == NODE_METRIC}
    assert node[("restorer", "dehaze")] == pytest.approx(0.9)  # confirmed replaces cheap 1.0
    assert node[("restorer", "none")] == pytest.approx(0.1)  # only a confirmed 0.1 row
    assert node[("sr", "off")] == pytest.approx(0.9)
    assert node[("detector_set", "a")] == pytest.approx(0.9)
    assert node[("detector_set", "a+b")] == pytest.approx(0.1)  # confirmed 0.1 replaces 0.667
    assert node[("detector_set", "b")] == 0.0  # never in the confirmed top three


def test_harvest_stores_raw_configuration_rows() -> None:
    raw = [r for r in run_harvest()[0] if r.node == "configuration"]
    assert len(raw) == 12
    assert {r.metric_name for r in raw} == {"f1_adjudicated", "f1_fused"}
    assert sum(r.metric_name == "f1_adjudicated" for r in raw) == 3
    assert all(r.source == "benchmark" and r.memory_version == "v1" for r in raw)
    assert "dehaze|off|a" in {r.option for r in raw}


def test_missing_cached_output_raises() -> None:
    with pytest.raises(ValueError, match="no cached detector output"):
        harvest_image(
            "x", profile(), "detect", blob_image(), [], {}, "dehaze", ["a"], GROUPING,
            lambda c: 0.0, 3, "v1",
        )  # fmt: skip


# ---- aggregation and versioning ---------------------------------------------------------------


def rec(key: str, node: str, option: str, value: float, sample: str = "s") -> ExperienceRecord:
    return ExperienceRecord.model_validate(
        {
            "profile_key": key,
            "query_type": "detect",
            "node": node,
            "option": option,
            "metric_name": NODE_METRIC,
            "metric_value": value,
            "sample_id": sample,
            "source": "benchmark",
            "memory_version": "v1",
        }
    )


def test_aggregate_mean_std_count_and_ignores_raw_rows() -> None:
    key = profile().key
    raw = ExperienceRecord.model_validate(
        {**rec(key, "restorer", "dehaze", 0.0).model_dump(), "node": "configuration"}
    )
    stats = aggregate(
        [rec(key, "restorer", "dehaze", 0.4), rec(key, "restorer", "dehaze", 0.6), raw]
    )
    assert len(stats) == 1
    s = stats[0]
    assert (s.mean, s.count) == (pytest.approx(0.5), 2)
    assert s.std == pytest.approx(0.1)


def test_memory_build_write_load_roundtrip(tmp_path: Path) -> None:
    records = [rec(profile().key, "restorer", "dehaze", 0.5)]
    created = datetime(2026, 10, 3, tzinfo=UTC)
    memory = build_memory(records, "v7", created)
    assert memory.version.record_count == 1
    assert memory.version.created == created
    assert memory.version.source_hash == build_memory(records, "v8", created).version.source_hash
    other = build_memory([rec(profile().key, "restorer", "none", 0.5)], "v7", created)
    assert other.version.source_hash != memory.version.source_hash
    path = write_memory(memory, tmp_path)
    assert path == memory_path(tmp_path, "v7")
    assert load_memory(path) == memory


# ---- retrieval --------------------------------------------------------------------------------

W = SETTINGS.thresholds.experience.similarity_weights


def test_similarity_zero_for_different_scene_labels() -> None:
    assert profile_similarity(profile("fog").key, profile("rain").key, W) == 0.0


def test_similarity_identical_is_one_and_ordinal() -> None:
    k = profile().key
    assert profile_similarity(k, k, W) == pytest.approx(1.0)
    near = profile_similarity(k, profile(visibility="moderate").key, W)
    far = profile_similarity(k, profile(visibility="clear").key, W)
    assert 1.0 > near > far > 0.0
    assert far == pytest.approx(1.0 - W.visibility / sum(W.model_dump().values()))


def test_similarity_respects_weights_and_mixed_scale() -> None:
    zero_vis = SimilarityWeights(illumination=1, visibility=0, object_scale=0, object_density=0)
    assert profile_similarity(profile().key, profile(visibility="clear").key, zero_vis) == 1.0
    mixed = profile_similarity(profile().key, profile(object_scale="mixed").key, W)
    assert mixed == pytest.approx(1.0 - 0.5 * W.object_scale / sum(W.model_dump().values()))


def make_memory() -> Memory:
    fog = profile()
    records = [
        rec(fog.key, "restorer", "dehaze", 0.52),
        rec(fog.key, "restorer", "dehaze", 0.52),
        rec(fog.key, "restorer", "none", 0.47),
        rec(fog.key, "restorer", "none", 0.47),
        rec(profile(visibility="moderate").key, "restorer", "dehaze", 0.60),
        rec(profile(visibility="moderate").key, "restorer", "none", 0.40),
        rec(profile("rain").key, "restorer", "derain", 0.9),
        rec(fog.key, "sr", "auto", 0.5),
    ]
    return build_memory(records, "v1", datetime(2026, 10, 3, tzinfo=UTC))


def test_retrieve_top_k_excludes_other_scenes_and_pools_options() -> None:
    r = retrieve(make_memory(), profile().key, "detect", EXP)
    assert [k for k, _ in r.profiles] == [profile().key, profile(visibility="moderate").key]
    dehaze, none = r.recommendations["restorer"]
    assert (dehaze.option, dehaze.count) == ("dehaze", 3)
    assert dehaze.mean == pytest.approx((0.52 * 2 + 0.60) / 3)
    assert none.option == "none"
    assert "derain" not in {o.option for o in r.recommendations["restorer"]}


def test_retrieve_respects_top_k_setting() -> None:
    one = EXP.model_copy(update={"top_k_profiles": 1})
    assert len(retrieve(make_memory(), profile().key, "detect", one).profiles) == 1


def test_retrieve_other_query_type_is_empty() -> None:
    r = retrieve(make_memory(), profile().key, "count", EXP)
    assert r.profiles == ()


# ---- injection --------------------------------------------------------------------------------


def test_render_golden_text() -> None:
    fog = profile()
    records = [rec(fog.key, "restorer", "dehaze", 0.52)] * 14 + [
        rec(fog.key, "restorer", "none", 0.47)
    ] * 14
    r = retrieve(build_memory(records, "v1"), fog.key, "detect", EXP)
    assert render_node(r, "restorer") == (
        "Similar scenes (1): dehaze F1 0.52 (n=14) vs none 0.47 (n=14)"
    )
    assert render_node(r, "sr") == ""
    assert render_all(r) == {
        "restorer": "Similar scenes (1): dehaze F1 0.52 (n=14) vs none 0.47 (n=14)",
        "sr": "",
        "detector_set": "",
    }


def test_render_empty_memory_is_empty_string() -> None:
    r = retrieve(build_memory([], "v0"), profile().key, "detect", EXP)
    assert render_all(r) == {"restorer": "", "sr": "", "detector_set": ""}


# ---- integration with restorer_select ---------------------------------------------------------

REPLY = '{"restorer": "dehaze", "rationale": "haze"}'


def prompt_for(memory: Memory) -> str:
    p = profile()
    text = render_node(retrieve(memory, p.key, "detect", EXP), "restorer")
    vlm = FakeVLM([REPLY])
    restorer_select(vlm, TraceCollector(), p, experience=text)
    return vlm.prompts[0]


def test_experience_table_reaches_restorer_prompt_and_empty_memory_is_noop() -> None:
    with_memory = prompt_for(make_memory())
    assert "Similar scenes (2): dehaze F1 0.55 (n=3) vs none 0.45 (n=3)" in with_memory

    empty = prompt_for(build_memory([], "v0"))
    vlm = FakeVLM([REPLY])
    restorer_select(vlm, TraceCollector(), profile())  # DetAS: no experience argument
    assert empty == vlm.prompts[0]
    assert "Similar scenes" not in empty
