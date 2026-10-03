"""Experience record and its JSONL store (system-design 5.7)."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

Node = Literal["restorer", "sr", "detector_set"]
# "configuration" rows hold raw per-configuration results; aggregation ignores them.
RecordNode = Literal["restorer", "sr", "detector_set", "configuration"]
Source = Literal["benchmark", "feedback"]

NODE_METRIC = "f1_best"  # node-level value: best F1 achievable given the option


class ExperienceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_key: str
    query_type: str
    node: RecordNode
    option: str
    metric_name: str
    metric_value: float
    sample_id: str
    source: Source
    memory_version: str


class ExperienceStore:
    """Append-only JSONL file of experience records."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, records: Iterable[ExperienceRecord]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")
                count += 1
        return count

    def load(self) -> list[ExperienceRecord]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [ExperienceRecord.model_validate_json(line) for line in f if line.strip()]
