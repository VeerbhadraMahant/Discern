"""Instantiate the adapter class named by a registry entry."""

import importlib

from discern.models.manager import RegistryEntry


def load_adapter(entry: RegistryEntry, device: str = "cuda") -> object:
    module_path, _, class_name = entry.adapter.rpartition(".")
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(entry, device=device)
