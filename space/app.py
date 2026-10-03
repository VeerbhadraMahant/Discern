"""Hugging Face Space entry point. The bundle keeps the repository layout (src/, configs/,
legacy/v0/), so the package is imported from ./src and its config paths resolve unchanged."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from discern.serve.gradio_app import main  # noqa: E402

main()
