"""query_parse node: question plus summaries of earlier result sets in, QueryPlan out
(system-design 5.5 and 5.6). References are checked in code; an unresolvable one yields a
clarifying question instead of a guess."""

from collections.abc import Sequence

from pydantic import BaseModel

from discern.agent.llm_io import load_prompt, structured_call
from discern.models.roles import VLM
from discern.query.schemas import QueryPlan, ResultSetSummary
from discern.trace import TraceCollector


class Clarification(BaseModel):
    message: str


def _format(summary: ResultSetSummary) -> str:
    span = (
        f"{summary.time_range.start:.1f}s to {summary.time_range.end:.1f}s"
        if summary.time_range
        else "none"
    )
    return (
        f'{summary.id}: question="{summary.question}", targets={", ".join(summary.targets)}, '
        f'count={summary.count}, time span={span}, aliases={", ".join(summary.aliases) or "none"}'
    )


def _references(plan: QueryPlan) -> list[str]:
    refs = [plan.source_result_set] if plan.source_result_set else []
    refs += [o.value for o in plan.operations if o.kind == "compare" and o.value]
    return refs


def query_parse(
    vlm: VLM,
    trace: TraceCollector,
    question: str,
    summaries: Sequence[ResultSetSummary],
) -> QueryPlan | Clarification:
    """Parse a question into a plan. Fallback: a describe query carrying the raw question.

    A refine without a source, or any reference to a result set that does not exist, returns a
    `Clarification` listing the available results.
    """
    results = "\n".join(_format(s) for s in summaries) or "none"
    plan = structured_call(
        vlm,
        load_prompt("query_parse"),
        {"question": question, "results": results},
        QueryPlan,
        lambda: QueryPlan(query_type="describe"),
        trace,
    ).model_copy(update={"question": question})
    known = {s.id for s in summaries}
    unresolved = [r for r in _references(plan) if r not in known]
    if unresolved or (plan.query_type == "refine" and not plan.source_result_set):
        message = (
            "Which earlier result do you mean? Available results: "
            + ("; ".join(_format(s) for s in summaries) or "none yet, so ask a new question first")
            + "."
        )
        with trace.span("query_parse.guard") as span:
            span.input_summary = f"question={question[:60]}, references={_references(plan)}"
            span.decision = "clarification"
            span.rationale = f"unresolved result set reference: {unresolved or 'missing source'}"
        return Clarification(message=message)
    return plan
