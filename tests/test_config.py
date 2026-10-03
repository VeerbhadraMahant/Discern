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
    assert t.experience.top_k_profiles == 3
    assert t.experience.harvest_samples_per_dataset == 50


def test_profile_selected_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCERN_PROFILE", "hosted_full")
    assert load_settings().profile.name == "hosted_full"


def test_config_hash_is_stable_and_profile_specific() -> None:
    assert load_settings("local_lite").config_hash == load_settings("local_lite").config_hash
    assert load_settings("local_lite").config_hash != load_settings("hosted_full").config_hash
