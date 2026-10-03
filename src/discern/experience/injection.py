"""Render retrieved experience as the compact text passed to the SAIR nodes' `experience` param."""

from collections.abc import Sequence

from discern.config.settings import ExperienceThresholds
from discern.experience.aggregate import Memory
from discern.experience.retrieval import Retrieval, retrieve
from discern.experience.schema import Node

MAX_OPTIONS = 3
IMAGE_QUERY_TYPE = "detect"  # the query type the harvest records image decisions under


def render_node(retrieval: Retrieval, node: Node, max_options: int = MAX_OPTIONS) -> str:
    """One line such as 'Similar scenes (3): dehaze F1 0.52 (n=14) vs none 0.47 (n=14)'.

    Empty when there is nothing to report, so prompts equal those of DetAS without experience.
    """
    options = retrieval.recommendations.get(node, ())[:max_options]
    if not retrieval.profiles or not options:
        return ""
    cells = [f"{o.option} {o.mean:.2f} (n={o.count})" for o in options]
    parts = " vs ".join([cells[0].replace(" ", " F1 ", 1), *cells[1:]])
    return f"Similar scenes ({len(retrieval.profiles)}): {parts}"


def render_all(retrieval: Retrieval, max_options: int = MAX_OPTIONS) -> dict[Node, str]:
    nodes: tuple[Node, ...] = ("restorer", "sr", "detector_set")
    return {n: render_node(retrieval, n, max_options) for n in nodes}


def render_for(
    memory: Memory,
    profile_key: str,
    nodes: Sequence[Node],
    settings: ExperienceThresholds,
    query_type: str = IMAGE_QUERY_TYPE,
) -> str:
    """Retrieve experience for `profile_key` and render the lines of `nodes`, one per line."""
    found = retrieve(memory, profile_key, query_type, settings)
    return "\n".join(line for n in nodes if (line := render_node(found, n)))
