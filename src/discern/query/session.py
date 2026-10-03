"""One conversational turn: parse, plan, execute, answer, verify, and update the state
(system-design 5.5 and 5.6)."""

import dataclasses
from collections.abc import Sequence

from discern.agent.nodes.query_parse import Clarification, query_parse
from discern.index.video_index import Segment
from discern.query.answer import compose_answer
from discern.query.executors import Services, execute
from discern.query.schemas import ConversationState, QueryPlan, ResultSet, Turn
from discern.video.types import Track


def _reusing_scans(services: Services, state: ConversationState) -> Services:
    """Wrap the caller's track provider: tracks get conversation-unique ids, and a full scan of
    the same targets is run once per conversation."""
    raw = services.detect

    def detect(targets: Sequence[str], segments: Sequence[Segment] | None) -> list[Track]:
        key = tuple(sorted(t.lower() for t in targets))
        if key in state.scan_cache:
            return state.scan_cache[key]
        tracks = state.adopt(raw(targets, segments))
        if segments is None:
            state.scan_cache[key] = tracks
        return tracks

    return dataclasses.replace(services, detect=detect)


def _alias(plan: QueryPlan, state: ConversationState) -> str | None:
    """A phrase for a refined set, so "the red ones" can be resolved later."""
    for op in plan.operations:
        if op.kind == "filter_attribute" and op.value and plan.source_result_set:
            targets = state.result_sets[plan.source_result_set].facts.targets
            return f"{op.value} {' '.join(targets)}".strip().lower()
    return None


def answer_question(
    state: ConversationState, question: str, services: Services
) -> tuple[ResultSet | None, str]:
    """Answer one question and record it in `state`.

    Returns the new result set and the verified answer text. When the question refers to an
    earlier result that does not exist, nothing runs: the result is None and the text is a
    clarifying question.
    """
    parsed = query_parse(services.vlm, services.trace, question, state.summaries())
    if isinstance(parsed, Clarification):
        state.turns.append(Turn(question=question, answer=parsed.message, result_set_id=None))
        return None, parsed.message
    execution = execute(parsed, _reusing_scans(services, state), state)
    text = compose_answer(
        services.vlm,
        services.trace,
        question,
        execution.facts,
        execution.context,
        services.settings,
    )
    result = ResultSet(
        id=state.new_result_id(),
        question=question,
        plan=parsed,
        tracks=execution.tracks,
        facts=execution.facts,
        answer=text,
        parent=parsed.source_result_set,
    )
    state.result_sets[result.id] = result
    if alias := _alias(parsed, state):
        state.aliases[alias] = result.id
    state.turns.append(Turn(question=question, answer=text, result_set_id=result.id))
    return result, text
