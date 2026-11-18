"""Settings shared by the example scripts."""

from __future__ import annotations

import os

MODEL_VARIABLE = "MODEL"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def example_model() -> str:
    """Return the model the examples use: ``$MODEL``, or a small OpenAI model.

    Set ``MODEL`` to try another vendor without editing a script, e.g.
    ``MODEL=anthropic/claude-sonnet-4-6``.
    """
    return os.environ.get(MODEL_VARIABLE, DEFAULT_MODEL)
