"""Aggregation of node-level records into a versioned memory (system-design 5.7 step 6)."""

import hashlib
import importlib
import math
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from discern.experience.schema import NODE_METRIC, ExperienceRecord, Node


class OptionStat(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_key: str
    query_type: str
    node: Node
    option: str
    mean: float
    std: float  # population standard deviation
    count: int


class ConfigStat(BaseModel):
    """Raw per-configuration F1 per (profile key, query type): the object a joint policy ranks."""

    model_config = ConfigDict(frozen=True)

    profile_key: str
    query_type: str
    configuration: str  # label "restorer|sr|detector_set", e.g. "lowlight|auto|a+b"
    mean: float
    std: float  # population standard deviation
    count: int


class MemoryVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    created: datetime
    record_count: int
    source_hash: str  # sha256 over the sorted record lines the memory was built from


class Memory(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: MemoryVersion
    stats: tuple[OptionStat, ...]
    # Empty in memory files written before configuration stats existed (backward compatible).
    config_stats: tuple[ConfigStat, ...] = ()


def aggregate(records: Iterable[ExperienceRecord]) -> list[OptionStat]:
    """Mean, standard deviation and count of node-level values per
    (profile key, query type, node, option). Raw per-configuration rows are ignored."""
    groups: dict[tuple[str, str, Node, str], list[float]] = defaultdict(list)
    for r in records:
        if r.node != "configuration" and r.metric_name == NODE_METRIC:
            groups[(r.profile_key, r.query_type, r.node, r.option)].append(r.metric_value)
    stats: list[OptionStat] = []
    for (profile_key, query_type, node, option), values in sorted(groups.items()):
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        stats.append(
            OptionStat(
                profile_key=profile_key,
                query_type=query_type,
                node=node,
                option=option,
                mean=mean,
                std=math.sqrt(var),
                count=len(values),
            )
        )
    return stats


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def aggregate_configurations(records: Iterable[ExperienceRecord]) -> list[ConfigStat]:
    """Mean, standard deviation and count of the raw per-configuration F1 (the "configuration"
    rows, whether cheap-fused or adjudicated) per (profile key, query type, configuration)."""
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in records:
        if r.node == "configuration":
            groups[(r.profile_key, r.query_type, r.option)].append(r.metric_value)
    out: list[ConfigStat] = []
    for (profile_key, query_type, label), values in sorted(groups.items()):
        mean, std = _mean_std(values)
        out.append(
            ConfigStat(
                profile_key=profile_key,
                query_type=query_type,
                configuration=label,
                mean=mean,
                std=std,
                count=len(values),
            )
        )
    return out


def with_configuration_stats(memory: Memory, records: Iterable[ExperienceRecord]) -> Memory:
    """The same memory version with configuration stats rebuilt from its raw records."""
    return memory.model_copy(update={"config_stats": tuple(aggregate_configurations(records))})


def build_memory(
    records: list[ExperienceRecord], version_id: str, created: datetime | None = None
) -> Memory:
    lines = sorted(r.model_dump_json() for r in records)
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    version = MemoryVersion(
        id=version_id,
        created=created or datetime.now(UTC),
        record_count=len(records),
        source_hash=digest,
    )
    return Memory(
        version=version,
        stats=tuple(aggregate(records)),
        config_stats=tuple(aggregate_configurations(records)),
    )


def memory_path(directory: Path, version_id: str) -> Path:
    return directory / f"memory-{version_id}.json"


def write_memory(memory: Memory, directory: Path) -> Path:
    path = memory_path(directory, memory.version.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(memory.model_dump_json(indent=1), encoding="utf-8", newline="\n")
    return path


def load_memory(path: Path) -> Memory:
    return Memory.model_validate_json(path.read_text(encoding="utf-8"))


def publish_to_hf(version: MemoryVersion, directory: Path, repo: str) -> str:
    """Upload one memory version file to a Hugging Face dataset repository.

    Needs network and credentials; `huggingface_hub` is imported here so core code never
    depends on it. Returns the path of the file inside the repository.
    """
    hub = importlib.import_module("huggingface_hub")
    path_in_repo = f"memory/{memory_path(directory, version.id).name}"
    hub.HfApi().upload_file(
        path_or_fileobj=str(memory_path(directory, version.id)),
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"memory version {version.id}",
    )
    return path_in_repo
