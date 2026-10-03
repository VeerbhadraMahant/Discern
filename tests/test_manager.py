import pytest

from discern.models.manager import ModelManager, ModelTooLargeError, RegistryEntry


def entry(name: str, role: str, vram_gb: float) -> RegistryEntry:
    return RegistryEntry(
        name=name,
        role=role,
        model_id=f"org/{name}",
        revision="rev",
        license="test",
        adapter="tests.Fake",
        vram_gb=vram_gb,
    )


REGISTRY = {
    "vlm": entry("vlm", "agent_vlm", 5.0),
    "det_a": entry("det_a", "detector", 1.0),
    "det_b": entry("det_b", "detector", 1.5),
    "huge": entry("huge", "agent_vlm", 20.0),
}
ACTIVE = {"agent_vlm": "vlm", "detector": "det_a"}


def make(budget_gb: float = 7.0) -> tuple[ModelManager, list[str], list[object]]:
    loads: list[str] = []
    unloads: list[object] = []

    def loader(e: RegistryEntry) -> object:
        loads.append(e.name)
        return f"model:{e.name}"

    return ModelManager(REGISTRY, ACTIVE, budget_gb, loader, unloads.append), loads, unloads


def test_get_uses_active_entry_and_loads_lazily_once() -> None:
    mgr, loads, _ = make()
    assert mgr.get("detector") == "model:det_a"
    assert mgr.get("detector") == "model:det_a"
    assert loads == ["det_a"]


def test_get_by_explicit_name() -> None:
    mgr, _, _ = make()
    assert mgr.get("detector", "det_b") == "model:det_b"


def test_role_mismatch_rejected() -> None:
    mgr, _, _ = make()
    with pytest.raises(ValueError):
        mgr.get("detector", "vlm")


def test_evicts_least_recently_used_when_over_budget() -> None:
    mgr, _, unloads = make(budget_gb=7.0)
    mgr.get("agent_vlm")  # 5.0
    mgr.get("detector", "det_a")  # 6.0
    mgr.get("agent_vlm")  # touch: det_a is now least recently used
    mgr.get("detector", "det_b")  # 5.0 + 1.5 fits in 7.0 only after evicting det_a
    assert unloads == ["model:det_a"]
    assert mgr.loaded_names == ["vlm", "det_b"]
    assert mgr.used_gb == pytest.approx(6.5)


def test_evicts_several_models_if_needed() -> None:
    mgr, _, unloads = make(budget_gb=7.0)
    mgr.get("detector", "det_a")
    mgr.get("detector", "det_b")
    mgr.get("agent_vlm")  # 5.0 needs 2.5 + 5.0 > 7.0, so evict det_a then fit
    assert unloads == ["model:det_a"]
    assert mgr.loaded_names == ["det_b", "vlm"]


def test_model_larger_than_budget_is_rejected_without_evicting() -> None:
    mgr, _, unloads = make()
    mgr.get("detector")
    with pytest.raises(ModelTooLargeError):
        mgr.get("agent_vlm", "huge")
    assert unloads == []
    assert mgr.loaded_names == ["det_a"]


def test_evict_unloads_now_and_is_a_noop_when_not_loaded() -> None:
    mgr, loads, unloads = make()
    mgr.get("detector")
    mgr.evict("det_b")  # not loaded
    assert unloads == []
    mgr.evict("det_a")
    assert unloads == ["model:det_a"] and mgr.loaded_names == []
    mgr.get("detector")  # loads again after an explicit eviction
    assert loads == ["det_a", "det_a"]
