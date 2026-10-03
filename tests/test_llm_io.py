from pathlib import Path

import pytest
from pydantic import BaseModel

from discern.agent.llm_io import load_prompt, structured_call
from discern.models.fakes import FakeVLM
from discern.trace import TraceCollector

HERE = Path(__file__).parent


class Choice(BaseModel):
    pick: str
    rationale: str = ""


FALLBACK = Choice(pick="original", rationale="fallback")
VALID = '{"pick": "restored", "rationale": "clearer"}'


def run(vlm: FakeVLM) -> tuple[Choice, TraceCollector]:
    trace = TraceCollector()
    prompt = load_prompt("example", 1, HERE / "prompts")
    out = structured_call(
        vlm, prompt, {"scene": "fog", "options": "A, B"}, Choice, lambda: FALLBACK, trace
    )
    return out, trace


def test_valid_output_uses_one_call() -> None:
    vlm = FakeVLM([VALID])
    out, trace = run(vlm)
    assert out.pick == "restored"
    assert len(vlm.prompts) == 1
    (event,) = trace.events
    assert event.node == "example"
    assert event.prompt_version == "example@v1"
    assert event.fallback_used is False


def test_code_fenced_json_is_accepted() -> None:
    out, _ = run(FakeVLM([f"```json\n{VALID}\n```"]))
    assert out.pick == "restored"


def test_invalid_then_valid_repairs_once() -> None:
    vlm = FakeVLM(["not json", VALID])
    out, trace = run(vlm)
    assert out.pick == "restored"
    assert len(vlm.prompts) == 2
    assert "invalid" in vlm.prompts[1] and "not json" in vlm.prompts[1]
    assert trace.events[0].fallback_used is False


def test_invalid_twice_falls_back() -> None:
    vlm = FakeVLM(["nope", '{"wrong": 1}'])
    out, trace = run(vlm)
    assert out == FALLBACK
    assert len(vlm.prompts) == 2
    assert trace.events[0].fallback_used is True


def test_load_prompt_picks_latest_version_by_default() -> None:
    assert load_prompt("example", directory=HERE / "prompts").version == 2
    with pytest.raises(FileNotFoundError):
        load_prompt("missing", directory=HERE / "prompts")


def test_rendered_prompt_matches_golden_snapshot() -> None:
    rendered = load_prompt("example", 1, HERE / "prompts").render(scene="fog", options="A, B")
    expected = (HERE / "snapshots" / "example.v1.txt").read_text(encoding="utf-8")
    assert rendered == expected
