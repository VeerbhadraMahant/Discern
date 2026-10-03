"""Query layer data model (system-design 5.5, 5.6 and 6): plans, facts, result sets and the
conversation state that follow-up questions operate on."""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from discern.video.types import Track
from discern.vision.boxes import Box

QueryType = Literal["locate", "count", "temporal", "relation", "describe", "refine"]
Predicate = Literal["near", "left_of", "right_of", "above", "below", "overlapping"]
OperationKind = Literal[
    "filter_attribute", "filter_time", "filter_region", "count", "show", "compare"
]
Region = Literal["left", "right", "top", "bottom"]


class TimeRange(BaseModel):
    start: float
    end: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError("time range end is before its start")
        return self


class Relation(BaseModel):
    subject: str
    predicate: Predicate
    object: str


class AttributeConstraint(BaseModel):
    target: str
    attribute: str  # for example "color"
    value: str  # for example "red"


class Operation(BaseModel):
    """One refine step. The fields each kind needs are checked on construction."""

    kind: OperationKind
    attribute: str | None = None
    value: str | None = None  # attribute value, or the other result set id for compare
    time_range: TimeRange | None = None
    region: Region | None = None

    @model_validator(mode="after")
    def _has_arguments(self) -> Self:
        needs = {
            "filter_attribute": self.attribute and self.value,
            "filter_time": self.time_range,
            "filter_region": self.region,
            "compare": self.value,
        }
        if self.kind in needs and not needs[self.kind]:
            raise ValueError(f"operation {self.kind} is missing its arguments")
        return self


class QueryPlan(BaseModel):
    query_type: QueryType
    question: str = ""  # the user's raw question, set by code
    targets: list[str] = Field(default_factory=list)
    attributes: list[AttributeConstraint] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    time_range: TimeRange | None = None
    source_result_set: str | None = None
    operations: list[Operation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _complete(self) -> Self:
        t = self.query_type
        if t == "refine" and not self.operations:
            raise ValueError("refine needs operations")
        if t in ("locate", "count", "temporal") and not (self.targets or self.source_result_set):
            raise ValueError(f"{t} needs targets or a source result set")
        if t == "relation" and not self.relations:
            raise ValueError("relation needs at least one relation")
        return self


class TrackFact(BaseModel):
    track_id: int
    label: str
    t_start: float
    t_end: float
    boxes: dict[int, Box]  # sampled frame index -> original-frame box


class PairFact(BaseModel):
    subject_track: int
    object_track: int
    time_ranges: list[TimeRange]  # when the relation holds


class Facts(BaseModel):
    """Everything an answer may state, computed by code. `grounded` is false only for describe."""

    query_type: QueryType
    grounded: bool = True
    targets: list[str] = Field(default_factory=list)
    count: int | None = None
    tracks: list[TrackFact] = Field(default_factory=list)
    time_ranges: list[TimeRange] = Field(default_factory=list)
    pairs: list[PairFact] = Field(default_factory=list)
    relation: str | None = None  # for example "person left_of car"
    compared: dict[str, int] = Field(default_factory=dict)  # result set id -> track count


class ResultSet(BaseModel):
    """The accepted tracks and evidence produced by answering one question."""

    id: str
    question: str
    plan: QueryPlan
    tracks: list[Track] = Field(default_factory=list)
    facts: Facts
    answer: str
    parent: str | None = None  # source result set of a follow-up

    @property
    def track_ids(self) -> list[int]:
        return [t.id for t in self.tracks]

    @property
    def grounded(self) -> bool:
        return self.facts.grounded


class ResultSetSummary(BaseModel):
    """Compact view of a result set shown to `query_parse` for reference resolution."""

    id: str
    question: str
    targets: list[str]
    count: int | None
    time_range: TimeRange | None
    aliases: list[str]


class Turn(BaseModel):
    question: str
    answer: str
    result_set_id: str | None  # None when the turn was a clarification


class ConversationState(BaseModel):
    turns: list[Turn] = Field(default_factory=list)
    result_sets: dict[str, ResultSet] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)  # phrase -> result set id
    # (track id, attribute) -> classified value. Track ids are unique across the conversation.
    attribute_cache: dict[tuple[int, str], str] = Field(default_factory=dict)
    # sorted targets -> tracks from a full scan, so a repeated scan does not re-run detection
    scan_cache: dict[tuple[str, ...], list[Track]] = Field(default_factory=dict)
    next_track_id: int = 1

    def adopt(self, tracks: list[Track]) -> list[Track]:
        """Renumber freshly computed tracks so ids stay unique across result sets."""
        adopted = []
        for track in tracks:
            adopted.append(track.model_copy(update={"id": self.next_track_id}))
            self.next_track_id += 1
        return adopted

    def new_result_id(self) -> str:
        return f"R{len(self.result_sets) + 1}"

    def summaries(self) -> list[ResultSetSummary]:
        out = []
        for rs in self.result_sets.values():
            starts = [t.t_start for t in rs.facts.tracks]
            ends = [t.t_end for t in rs.facts.tracks]
            out.append(
                ResultSetSummary(
                    id=rs.id,
                    question=rs.question,
                    targets=rs.facts.targets,
                    count=rs.facts.count,
                    time_range=TimeRange(start=min(starts), end=max(ends)) if starts else None,
                    aliases=[a for a, rid in self.aliases.items() if rid == rs.id],
                )
            )
        return out
