from pathlib import Path

import yaml

from discern.config.settings import CONFIG_DIR
from discern.models.manager import RegistryEntry


def load_registry(config_dir: Path = CONFIG_DIR) -> dict[str, RegistryEntry]:
    """Load `configs/models.yaml` into entries keyed by name."""
    with (config_dir / "models.yaml").open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {name: RegistryEntry(name=name, **fields) for name, fields in raw.items()}
