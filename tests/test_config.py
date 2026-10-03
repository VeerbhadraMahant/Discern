import pytest

from discern.config import load_settings


@pytest.mark.parametrize("name", ["hosted_full", "local_lite"])
def test_profiles_load(name: str) -> None:
    settings = load_settings(name)
    assert settings.profile.name == name


def test_paper_values() -> None:
    t = load_settings("local_lite").thresholds
    assert t.grouping.alpha == 0.25
    assert t.grouping.theta_iou == 0.5
    assert t.grouping.phi_vis == 0.5
    assert t.agent.top_k_detectors == 2
    assert t.agent.sr_target_long_side == 2048
    assert t.agent.fallback_accept_score == 0.3
    assert t.experience.top_k_profiles == 3
    assert t.experience.harvest_samples_per_dataset == 50
    assert t.experience.confirm_top_configs == 3
    w = t.experience.similarity_weights
    assert (w.illumination, w.visibility, w.object_scale, w.object_density) == (1.0, 1.0, 0.5, 0.5)


def test_video_thresholds() -> None:
    v = load_settings("local_lite").thresholds.video
    assert v.min_shot_seconds == 1.0
    assert 0.0 < v.reid_cosine <= 1.0
    assert v.reid_max_gap_seconds > 0


def test_profile_selected_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCERN_PROFILE", "hosted_full")
    assert load_settings().profile.name == "hosted_full"


def test_config_hash_is_stable_and_profile_specific() -> None:
    assert load_settings("local_lite").config_hash == load_settings("local_lite").config_hash
    assert load_settings("local_lite").config_hash != load_settings("hosted_full").config_hash


def test_index_and_query_thresholds() -> None:
    t = load_settings("local_lite").thresholds
    assert t.index.smoothing_seconds > 0 and t.index.top_segments >= 1
    assert t.query.time_tolerance_seconds > 0
    assert 0.0 < t.query.region_min_fraction <= 1.0


def test_serve_and_monitor_thresholds() -> None:
    t = load_settings("local_lite").thresholds
    assert t.serve.ttl_seconds > 0 and t.serve.max_upload_mb > 0
    assert 0.0 < t.monitor.drift_similarity <= 1.0
