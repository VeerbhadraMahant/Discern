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
    fallback_accept_score: float
    default_operating_threshold: float
    adjudicate_min_context_px: int
    adjudicate_min_view_px: int


class SimilarityWeights(_Frozen):
    illumination: float
    visibility: float
    object_scale: float
    object_density: float


class ExperienceThresholds(_Frozen):
    top_k_profiles: int
    harvest_samples_per_dataset: int
    confirm_top_configs: int
    similarity_weights: SimilarityWeights


class VideoThresholds(_Frozen):
    min_shot_seconds: float
    cut_threshold: float
    track_activation_score: float
    track_lost_seconds: float
    reid_cosine: float
    reid_max_gap_seconds: float
    duration_margin_seconds: float


class IndexThresholds(_Frozen):
    smoothing_seconds: float
    top_segments: int


class QueryThresholds(_Frozen):
    time_tolerance_seconds: float
    near_ratio: float
    direction_ratio: float
    overlap_ratio: float
    region_min_fraction: float


class DegradeThresholds(_Frozen):
    """Synthetic degradation ranges, each (value at severity 0, value at severity 1)."""

    fog_beta: tuple[float, float]  # scattering coefficient over normalised depth
    fog_airlight: tuple[float, float]  # atmospheric light, 0..1
    low_light_gamma: tuple[float, float]
    low_light_scale: tuple[float, float]
    low_light_peak: tuple[float, float]  # photons at full scale: lower means more shot noise
    rain_density: tuple[float, float]  # streaks per pixel
    rain_length: tuple[float, float]  # streak length as a fraction of frame height
    rain_brightness: tuple[float, float]  # streak opacity, 0..1
    rain_angle_degrees: float  # streak tilt from vertical
    rain_fall_fraction: float  # per-frame streak fall as a fraction of frame height
    noise_sigma: tuple[float, float]  # gaussian sigma in 8-bit levels
    noise_peak: tuple[float, float]  # poisson photons at full scale


class EvalGateThresholds(_Frozen):
    """Gate: a higher-is-better metric must reach its minimum, a lower-is-better metric must stay
    under its maximum, each within a relative tolerance."""

    experiment: str  # MLflow experiment holding the gate runs
    tolerance: float  # relative slack applied to every threshold
    minimums: dict[str, float]
    maximums: dict[str, float]


class GpuSeconds(_Frozen):
    """Duration requested for each GPU-decorated call (ZeroGPU `duration`)."""

    clean_image: int
    detect: int
    ingest: int
    index: int
    ask: int


class ServeThresholds(_Frozen):
    ttl_seconds: float  # session directory lifetime after last use
    max_upload_mb: float
    max_pixels: int  # largest accepted image, width x height
    memory_dir: str  # experience memory files and the pointer that pins the live version
    memory_pointer: str  # pointer file name inside memory_dir
    max_queries_per_session: int
    cleanup_interval_seconds: float  # how often expired sessions are swept
    gpu_seconds: GpuSeconds


class MonitorThresholds(_Frozen):
    drift_similarity: float  # best profile similarity below this counts an upload as drifted


class Thresholds(_Frozen):
    degrade: DegradeThresholds
    eval_gate: EvalGateThresholds
    grouping: GroupingThresholds
    agent: AgentThresholds
    experience: ExperienceThresholds
    video: VideoThresholds
    index: IndexThresholds
    query: QueryThresholds
    serve: ServeThresholds
    monitor: MonitorThresholds


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
