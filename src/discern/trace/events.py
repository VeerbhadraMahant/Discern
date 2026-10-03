"""Trace events: every pipeline node records what it decided, why, and what it cost."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    node: str
    started_at: datetime
    duration_ms: float
    gpu_ms: float = 0.0
    model_versions: dict[str, str] = Field(default_factory=dict)
    prompt_version: str | None = None
    input_summary: str = ""
    decision: str = ""
    rationale: str = ""
    fallback_used: bool = False


class Span:
    """Mutable handle filled in by the node while it runs inside `TraceCollector.span`."""

    def __init__(self, node: str) -> None:
        self.node = node
        self.gpu_ms = 0.0
        self.model_versions: dict[str, str] = {}
        self.prompt_version: str | None = None
        self.input_summary = ""
        self.decision = ""
        self.rationale = ""
        self.fallback_used = False


class TraceCollector:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    @contextmanager
    def span(self, node: str) -> Iterator[Span]:
        span = Span(node)
        started_at = datetime.now(UTC)
        t0 = time.perf_counter()
        try:
            yield span
        finally:
            self.events.append(
                TraceEvent(
                    node=node,
                    started_at=started_at,
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                    gpu_ms=span.gpu_ms,
                    model_versions=span.model_versions,
                    prompt_version=span.prompt_version,
                    input_summary=span.input_summary,
                    decision=span.decision,
                    rationale=span.rationale,
                    fallback_used=span.fallback_used,
                )
            )
