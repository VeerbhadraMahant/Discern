"""Answer composition (system-design 5.5): the VLM phrases an answer from Facts only, a verifier
checks every number and timestamp in the text against Facts, and a deterministic template
replaces the text on any mismatch (Claude.md invariant 1)."""

import re

from discern.agent.llm_io import load_prompt, structured_call
from discern.agent.schemas import AnswerText
from discern.config.settings import Settings, load_settings
from discern.models.roles import VLM
from discern.query.schemas import Facts, TimeRange
from discern.trace import TraceCollector

UNGROUNDED_LABEL = "Ungrounded (based on shot captions, not on verified detections):"
VERIFY_NODE = "answer.verify"

_CLOCK = re.compile(r"\b(\d+):(\d{2})(?::(\d{2}))?(\.\d+)?")
_DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|s)\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?(?![A-Za-z0-9_])")
_WORDS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_TENS = "thirty forty fifty sixty seventy eighty ninety".split()
_WORDS |= {w: 10 * i for i, w in enumerate(_TENS, start=3)}
_WORDS |= {"hundred": 100, "none": 0, "both": 2, "couple": 2, "dozen": 12}
_WORD = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.IGNORECASE)
# a track reference ("track 3", "tracks 1, 2 and 3", "#3") names an id, not a quantity
_TRACK_REF = re.compile(
    r"(?:\btracks?\s*#?|#)\d+(?:\s*(?:,|and|&)\s*(?:and\s*)?#?\d+(?!\d|\.\d))*", re.IGNORECASE
)


def format_time(t: float) -> str:
    return f"{t:.1f}s"


def _span(r: TimeRange) -> str:
    return f"{format_time(r.start)} to {format_time(r.end)}"


def facts_summary(facts: Facts) -> str:
    """The facts as plain lines, for the answer prompt."""
    lines = [
        f"Query type: {facts.query_type}",
        f"Grounded: {'yes' if facts.grounded else 'no'}",
        f"Targets: {', '.join(facts.targets) or 'none'}",
    ]
    if facts.count is not None:
        lines.append(f"Count: {facts.count}")
    if facts.relation:
        lines.append(f"Relation: {facts.relation}")
    lines += [
        f"Track {t.track_id} ({t.label}): seen {_span(TimeRange(start=t.t_start, end=t.t_end))}"
        for t in facts.tracks
    ]
    lines += [
        f"Pair: track {p.subject_track} and track {p.object_track} while "
        + ", ".join(_span(r) for r in p.time_ranges)
        for p in facts.pairs
    ]
    lines += [f"Result {k}: {v} tracks" for k, v in facts.compared.items()]
    return "\n".join(lines)


def template_answer(facts: Facts, context: str = "") -> str:
    """Deterministic answer from Facts alone. Contains only numbers that Facts holds."""
    targets = " and ".join(facts.targets) or "objects"
    seen = "; ".join(
        f"track {t.track_id} ({t.label}) {_span(TimeRange(start=t.t_start, end=t.t_end))}"
        for t in facts.tracks
    )
    if facts.query_type == "describe":
        captions = context.replace("\n- ", " ").removeprefix("- ")
        return captions or "No shot captions are available, so I cannot describe the video."
    if facts.query_type == "relation":
        if not facts.pairs:
            return f"No pair matching {facts.relation or 'that relation'} was found."
        pairs = "; ".join(
            f"track {p.subject_track} and track {p.object_track} while "
            + ", ".join(_span(r) for r in p.time_ranges)
            for p in facts.pairs
        )
        return f"Found {len(facts.pairs)} pair(s) where {facts.relation}: {pairs}."
    if facts.query_type == "count":
        return f"I counted {facts.count} distinct {targets}." + (f" {seen}." if seen else "")
    if facts.query_type == "refine":
        text = f"{facts.count} track(s) of {targets} remain."
        if facts.compared:
            text += " Counts: " + ", ".join(f"{k} has {v}" for k, v in facts.compared.items()) + "."
        return text + (f" {seen}." if seen else "")
    if not facts.tracks:  # locate, temporal
        return f"No {targets} was found."
    return f"Found {len(facts.tracks)} object(s) matching {targets}: {seen}."


# ---- verifier ---------------------------------------------------------------------------------


def extract_numbers(text: str) -> tuple[list[float], list[float]]:
    """(times in seconds, plain numbers) found in `text`. Times are clock values like 1:05 and
    values with a unit like 3.2s or 4 seconds; plain numbers are digits and number words."""
    times: list[float] = []

    def clock(m: re.Match[str]) -> str:
        if m[3]:
            seconds = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
        else:
            seconds = int(m[1]) * 60 + int(m[2])
        times.append(seconds + float(m[4] or 0))
        return " "

    def duration(m: re.Match[str]) -> str:
        value = float(m[1])
        times.append(value * 60 if m[2].lower().startswith("m") else value)
        return " "

    rest = _DURATION.sub(duration, _CLOCK.sub(clock, text))
    numbers = [float(n) for n in _NUMBER.findall(rest)]
    numbers += [float(_WORDS[w.lower()]) for w in _WORD.findall(rest)]
    return times, numbers


def _allowed(facts: Facts) -> tuple[list[float], set[int], set[int]]:
    """(times, quantities, track ids) the answer may mention. A track id is only valid as a track
    reference, so it does not make the same number a valid count."""
    ranges = [TimeRange(start=t.t_start, end=t.t_end) for t in facts.tracks] + facts.time_ranges
    times = [x for r in ranges for x in (r.start, r.end, r.end - r.start)]
    quantities = {len(facts.tracks), *facts.compared.values()}
    if facts.pairs:  # no pairs must not make "zero" a valid claim
        quantities.add(len(facts.pairs))
    if facts.count is not None:
        quantities.add(facts.count)
    ids = {t.track_id for t in facts.tracks}
    ids |= {p.subject_track for p in facts.pairs} | {p.object_track for p in facts.pairs}
    return times, quantities, ids


def verify(text: str, facts: Facts, tolerance: float) -> list[str]:
    """Mismatches between the numbers and times in `text` and `facts`; empty means verified.
    An ungrounded answer has no facts, so it may contain no numbers or times at all."""
    allowed_times, allowed_numbers, allowed_ids = (
        _allowed(facts) if facts.grounded else ([], set(), set())
    )
    ids = [int(n) for ref in _TRACK_REF.findall(text) for n in re.findall(r"\d+", ref)]
    times, numbers = extract_numbers(_TRACK_REF.sub(" ", text))
    problems = [f"track {i} is not in the facts" for i in ids if i not in allowed_ids]
    problems += [
        f"time {t:g}s is not in the facts"
        for t in times
        if not any(abs(t - a) <= tolerance for a in allowed_times)
    ]
    problems += [
        f"number {n:g} is not in the facts"
        for n in numbers
        if not any(abs(n - a) < 1e-9 for a in allowed_numbers)
    ]
    return problems


def verifier_counts(trace: TraceCollector) -> tuple[int, int]:
    """(passed, fell back to the template) over all answers verified in this trace."""
    events = [e for e in trace.events if e.node == VERIFY_NODE]
    fallbacks = sum(e.fallback_used for e in events)
    return len(events) - fallbacks, fallbacks


def compose_answer(
    vlm: VLM,
    trace: TraceCollector,
    question: str,
    facts: Facts,
    context: str = "",
    settings: Settings | None = None,
) -> str:
    """Phrase an answer from Facts, verify it, and fall back to the template on any mismatch.
    Ungrounded answers carry `UNGROUNDED_LABEL`."""
    settings = settings or load_settings()
    template = template_answer(facts, context)
    text = structured_call(
        vlm,
        load_prompt("answer"),
        {"question": question, "facts": facts_summary(facts), "context": context or "none"},
        AnswerText,
        lambda: AnswerText(answer=template),
        trace,
    ).answer.strip()
    problems = verify(text, facts, settings.thresholds.query.time_tolerance_seconds)
    with trace.span(VERIFY_NODE) as span:
        span.input_summary = f"answer={text[:80]}"
        if problems:
            span.fallback_used = True
            span.decision = "fallback to template"
            span.rationale = "; ".join(problems)
            text = template
        else:
            span.decision = "pass"
    return text if facts.grounded else f"{UNGROUNDED_LABEL} {text}"
