from pathlib import Path

import pytest

from discern.experience.harvest import SR_AUTO, SR_OFF
from discern.experience.harvest_cache import (
    CacheKey,
    VariantCache,
    fill_cache,
    load_cached_outputs,
    mapped_restorer,
    pending,
    read_cache,
    scale_back,
    sr_factor,
    variant_keys,
    write_cache,
)
from discern.models.roles import Detection
from discern.vision.boxes import Box

TARGET = 2048


def det(x: float = 10.0, score: float = 0.5, name: str = "d") -> Detection:
    return Detection(box=Box(x, x, x + 20, x + 40), label="car", score=score, detector=name)


def test_mapped_restorer_from_scene_label() -> None:
    assert mapped_restorer("fog") == "dehaze"
    assert mapped_restorer("low_light") == "lowlight"
    assert mapped_restorer("rain") == "derain"
    assert mapped_restorer("normal") == "none"
    assert mapped_restorer(None) == "none"


def test_variant_keys() -> None:
    assert variant_keys("none") == [("none", SR_OFF), ("none", SR_AUTO)]
    assert variant_keys("dehaze") == [
        ("none", SR_OFF),
        ("none", SR_AUTO),
        ("dehaze", SR_OFF),
        ("dehaze", SR_AUTO),
    ]


def test_sr_factor_choice_and_skipping() -> None:
    assert sr_factor(1200, 800, TARGET) == 2  # 2048 / 1200 rounds up to 2
    assert sr_factor(1024, 700, TARGET) == 2
    assert sr_factor(640, 480, TARGET) == 4  # needs more than 2, clamped to 4
    assert sr_factor(480, 640, TARGET) == 4  # long side counts, not the width
    assert sr_factor(2048, 1000, TARGET) is None  # already reaches the target: auto equals off
    assert sr_factor(3000, 2000, TARGET) is None


def test_scale_back_to_original_frame() -> None:
    out = scale_back([det(40.0)], 4)
    assert out[0].box == Box(10.0, 10.0, 15.0, 20.0)
    assert out[0].label == "car" and out[0].score == 0.5
    same = [det()]
    assert scale_back(same, 1) == same


def test_cache_key_path(tmp_path: Path) -> None:
    key = CacheKey(tmp_path, "darkface", ("lowlight", SR_AUTO), "owlv2-base", "abcdef0123456789")
    assert key.path == tmp_path / "darkface" / "lowlight-auto-owlv2-base-abcdef0123.json"
    other = CacheKey(tmp_path, "darkface", ("lowlight", SR_AUTO), "owlv2-base", "ffffffffff0000")
    assert other.path != key.path  # a new model revision never reuses an old cache


def test_cache_round_trip(tmp_path: Path) -> None:
    key = CacheKey(tmp_path, "ds", ("none", SR_OFF), "d", "rev0123456789")
    assert read_cache(key) is None
    cache = VariantCache(detections={"a": [det(), det(50.0, 0.9)], "b": []}, skipped={"c"})
    write_cache(key, cache)
    back = read_cache(key)
    assert back is not None
    assert back.detections == cache.detections
    assert back.skipped == {"c"}


def test_cache_format_mismatch_is_rejected(tmp_path: Path) -> None:
    key = CacheKey(tmp_path, "ds", ("none", SR_OFF), "d", "rev")
    key.path.parent.mkdir(parents=True)
    key.path.write_text('{"format": 99, "detections": {}, "skipped": []}')
    with pytest.raises(ValueError, match="format"):
        read_cache(key)


def test_fill_cache_resumes_and_scales(tmp_path: Path) -> None:
    key = CacheKey(tmp_path, "ds", ("none", SR_AUTO), "d", "rev")
    calls: list[str] = []

    def detect(image_id: str) -> list[Detection]:
        calls.append(image_id)
        return [det(40.0)]

    todo: list[tuple[str, int | None]] = [("a", 4), ("b", None), ("c", 4)]
    fill_cache(key, todo[:1], detect)
    assert pending(key, ["a", "b", "c"]) == ["b", "c"]
    cache = fill_cache(key, todo, detect)
    assert calls == ["a", "c"]  # a is not recomputed, b is skipped without detecting
    assert cache.skipped == {"b"}
    assert cache.detections["c"][0].box == Box(10.0, 10.0, 15.0, 20.0)
    assert pending(key, ["a", "b", "c"]) == []


def test_load_cached_outputs_aliases_skipped_auto_to_off(tmp_path: Path) -> None:
    revisions = {"d1": "r1", "d2": "r2"}
    # image a: needs auto; image b already reaches the target, so auto is skipped for it.
    # Both images map to the dehaze restorer, so all four variants exist.
    for variant, factor in [
        (("none", SR_OFF), {"a": 1, "b": 1}),
        (("none", SR_AUTO), {"a": 2, "b": None}),
        (("dehaze", SR_OFF), {"a": 1, "b": 1}),
        (("dehaze", SR_AUTO), {"a": 2, "b": None}),
    ]:
        for name, rev in revisions.items():
            key = CacheKey(tmp_path, "ds", variant, name, rev)
            score = 0.1 if variant[0] == "none" else 0.8
            fill_cache(
                key, list(factor.items()), lambda _i, s=score, n=name: [det(score=s, name=n)]
            )
    out = load_cached_outputs(tmp_path, "ds", {"a": "dehaze", "b": "dehaze"}, revisions)
    assert set(out["a"]) == set(variant_keys("dehaze"))
    assert out["b"][("dehaze", SR_AUTO)]["d1"] == out["b"][("dehaze", SR_OFF)]["d1"]
    assert out["a"][("dehaze", SR_AUTO)]["d2"][0].detector == "d2"
    assert out["a"][("none", SR_OFF)]["d1"][0].score == 0.1


def test_load_cached_outputs_only_reads_needed_variants(tmp_path: Path) -> None:
    key_off = CacheKey(tmp_path, "ds", ("none", SR_OFF), "d", "r")
    key_auto = CacheKey(tmp_path, "ds", ("none", SR_AUTO), "d", "r")
    fill_cache(key_off, [("a", 1)], lambda _i: [det()])
    fill_cache(key_auto, [("a", None)], lambda _i: [det()])
    out = load_cached_outputs(tmp_path, "ds", {"a": "none"}, {"d": "r"})
    assert set(out["a"]) == {("none", SR_OFF), ("none", SR_AUTO)}


def test_load_cached_outputs_reports_missing_cache(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="harvest_variants"):
        load_cached_outputs(tmp_path, "ds", {"a": "none"}, {"d": "r"})
    key_off = CacheKey(tmp_path, "ds", ("none", SR_OFF), "d", "r")
    key_auto = CacheKey(tmp_path, "ds", ("none", SR_AUTO), "d", "r")
    fill_cache(key_off, [("a", 1)], lambda _i: [det()])
    fill_cache(key_auto, [("b", 2)], lambda _i: [det()])  # lacks image a
    with pytest.raises(ValueError, match="no cached"):
        load_cached_outputs(tmp_path, "ds", {"a": "none"}, {"d": "r"})
