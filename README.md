# Discern

Degradation-aware, grounded video query agent. Upload a video or image, Discern adaptively
cleans it, and you ask questions in natural language. Answers about finding, counting, timing
or relating objects point to boxes, tracks and timestamps you can inspect.

Built on the ideas of *Detect in Any Scene* (DetAS / DetAS-X, arXiv 2605.31174). Open models
only, runnable locally on an 8GB GPU and hosted on a Hugging Face ZeroGPU Space.

Status: under construction.

## Development

```
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```
