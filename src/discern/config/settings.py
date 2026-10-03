"""Typed settings loaded from YAML profiles and thresholds in `configs/`."""

import hashlib
import json
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
PROFILE_ENV_VAR = "DISCERN_PROFILE"
DEFAULT_PROFILE = "local_lite"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GroupingThresholds(_Frozen):
    alpha: float
    theta_iou: float
    phi_vis: float
    crop_size: int


class AgentThresholds(_Frozen):
    top_k_detectors: int
    sr_target_long_side: int


class ExperienceThresholds(_Frozen):
    top_k_profiles: int
    harvest_samples_per_dataset: int


class Thresholds(_Frozen):
    grouping: GroupingThresholds
    agent: AgentThresholds
    experience: ExperienceThresholds


class Profile(_Frozen):
    name: str
    vram_budget_gb: float
    max_video_seconds: int
    max_long_side_px: int
    sample_fps: float
    models: dict[str, str]  # role -> registry entry name


class Settings(_Frozen):
    profile: Profile
    thresholds: Thresholds

    @property
    def config_hash(self) -> str:
        """Stable short hash of the effective configuration, for evaluation records."""
        payload = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _read_yaml(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_settings(profile_name: str | None = None, config_dir: Path = CONFIG_DIR) -> Settings:
    """Load a profile (default: $DISCERN_PROFILE, else local_lite) plus thresholds."""
    name = profile_name or os.environ.get(PROFILE_ENV_VAR, DEFAULT_PROFILE)
    profile = Profile.model_validate(_read_yaml(config_dir / "profiles" / f"{name}.yaml"))
    thresholds = Thresholds.model_validate(_read_yaml(config_dir / "thresholds.yaml"))
    return Settings(profile=profile, thresholds=thresholds)
