from pathlib import Path

import pytest

from discern.config import load_settings
from discern.models.registry import load_registry

SRC = Path(__file__).resolve().parents[1] / "src"
REGISTRY = load_registry()


def test_every_entry_has_pinned_revision_and_license() -> None:
    for entry in REGISTRY.values():
        assert entry.revision and entry.license, entry.name


@pytest.mark.parametrize("profile", ["hosted_full", "local_lite"])
def test_profile_models_resolve_to_registry_entries_with_matching_role(profile: str) -> None:
    for role, name in load_settings(profile).profile.models.items():
        assert name in REGISTRY, f"{profile}: unknown model {name!r}"
        assert REGISTRY[name].role == role


@pytest.mark.parametrize("profile", ["hosted_full", "local_lite"])
def test_each_active_model_fits_the_profile_vram_budget(profile: str) -> None:
    settings = load_settings(profile)
    for name in settings.profile.models.values():
        assert REGISTRY[name].vram_gb <= settings.profile.vram_budget_gb, name


def test_no_hardcoded_model_ids_in_src() -> None:
    sources = "\n".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.py"))
    for entry in REGISTRY.values():
        assert entry.model_id not in sources, entry.model_id
