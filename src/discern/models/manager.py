"""Registry entries and the single owner of loaded models (`ModelManager`).

All model loading goes through the manager, which enforces the profile's VRAM budget and
evicts the least recently used model when a new one does not fit.
"""

import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class RegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    role: str
    model_id: str
    revision: str
    license: str
    speed_class: Literal["fast", "slow"] = "fast"
    adapter: str  # dotted path of the adapter class that wraps the model
    vram_gb: float  # approximate VRAM at this entry's precision; one entry per precision
    quantize_4bit: bool = False  # adapters load 4-bit (nf4) weights when set, else bf16


class ModelTooLargeError(RuntimeError):
    pass


class ModelManager:
    def __init__(
        self,
        registry: Mapping[str, RegistryEntry],
        active: Mapping[str, str],
        budget_gb: float,
        loader: Callable[[RegistryEntry], object],
        unloader: Callable[[object], None] = lambda model: None,
    ) -> None:
        self._registry = registry
        self._active = active  # role -> registry entry name, from the profile
        self._budget_gb = budget_gb
        self._loader = loader
        self._unloader = unloader
        self._loaded: OrderedDict[str, object] = OrderedDict()  # LRU order, oldest first

    @property
    def loaded_names(self) -> list[str]:
        return list(self._loaded)

    @property
    def used_gb(self) -> float:
        return sum(self._registry[n].vram_gb for n in self._loaded)

    def evict(self, name: str) -> None:
        """Unload a model now (no-op if not loaded), e.g. to free VRAM for the next phase."""
        model = self._loaded.pop(name, None)
        if model is not None:
            logger.info("evicting %s on request", name)
            self._unloader(model)

    def get(self, role: str, name: str | None = None) -> object:
        """Return the model for `role` (the profile's active entry unless `name` is given)."""
        name = name or self._active[role]
        entry = self._registry[name]
        if entry.role != role:
            raise ValueError(f"{name!r} has role {entry.role!r}, not {role!r}")
        if name in self._loaded:
            self._loaded.move_to_end(name)
            return self._loaded[name]
        if entry.vram_gb > self._budget_gb:
            raise ModelTooLargeError(
                f"{name} needs {entry.vram_gb} GB but the budget is {self._budget_gb} GB"
            )
        while self.used_gb + entry.vram_gb > self._budget_gb:
            victim, model = self._loaded.popitem(last=False)
            logger.info("evicting %s to fit %s", victim, name)
            self._unloader(model)
        model = self._loader(entry)
        self._loaded[name] = model
        return model
