"""Structured VLM calls: versioned prompt files, Pydantic validation, one repair retry,
deterministic fallback, and a trace event for every call."""

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template

from pydantic import BaseModel, ValidationError

from discern.models.roles import VLM, Image
from discern.trace import TraceCollector

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
_VERSION_RE = re.compile(r"^(?P<name>.+)\.v(?P<version>\d+)\.txt$")
_FENCE_RE = re.compile(r"^```(?:json)?\s*(?P<body>.*?)\s*```$", re.DOTALL)


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    template: str

    @property
    def tag(self) -> str:
        return f"{self.name}@v{self.version}"

    def render(self, **variables: object) -> str:
        return Template(self.template).substitute(**variables)


def load_prompt(name: str, version: int | None = None, directory: Path = PROMPTS_DIR) -> Prompt:
    """Load `<name>.v<version>.txt`; the highest version if `version` is not given."""
    versions = {}
    for path in directory.glob(f"{name}.v*.txt"):
        m = _VERSION_RE.match(path.name)
        if m and m["name"] == name:
            versions[int(m["version"])] = path
    if not versions:
        raise FileNotFoundError(f"no prompt named {name!r} in {directory}")
    chosen = version if version is not None else max(versions)
    return Prompt(name, chosen, versions[chosen].read_text(encoding="utf-8"))


def _schema_instruction(schema: type[BaseModel]) -> str:
    return (
        "\n\nRespond with a single JSON object and nothing else, matching this JSON schema:\n"
        + json.dumps(schema.model_json_schema())
    )


def _parse[T: BaseModel](text: str, schema: type[T]) -> T:
    text = text.strip()
    if m := _FENCE_RE.match(text):
        text = m["body"]
    return schema.model_validate_json(text)


def _generate(vlm: VLM, text: str, images: Sequence[Image]) -> str | None:
    """One VLM call. A runtime failure (CUDA out-of-memory is a RuntimeError) or a processor
    rejecting the input (extreme aspect-ratio crops raise ValueError) gives None."""
    try:
        return vlm.generate(text, images)
    except (RuntimeError, ValueError):
        logger.exception("VLM call failed")
        return None


def structured_call[T: BaseModel](
    vlm: VLM,
    prompt: Prompt,
    variables: dict[str, object],
    schema: type[T],
    fallback: Callable[[], T],
    trace: TraceCollector,
    images: Sequence[Image] = (),
) -> T:
    """Call the VLM and validate the output. One repair retry, then the deterministic fallback."""
    with trace.span(prompt.name) as span:
        span.prompt_version = prompt.tag
        span.input_summary = ", ".join(f"{k}={str(v)[:60]}" for k, v in variables.items())
        text = prompt.render(**variables) + _schema_instruction(schema)

        result: T | None = None
        raw = _generate(vlm, text, images)
        if raw is not None:
            try:
                result = _parse(raw, schema)
            except ValidationError as err:
                repair = (
                    f"{text}\n\nYour previous answer was invalid:\n{raw}\n\n"
                    f"Validation error:\n{err}\n\nReturn corrected JSON only."
                )
                raw = _generate(vlm, repair, images)
                if raw is not None:
                    try:
                        result = _parse(raw, schema)
                    except ValidationError:
                        pass
        if result is None:
            span.fallback_used = True
            result = fallback()
        span.decision = result.model_dump_json()
        return result
