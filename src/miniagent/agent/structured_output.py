"""Structured output: make an agent's final answer a validated Pydantic object.

Given an ``output_type``, the agent:

1. adds the type's JSON Schema to its system prompt, asking for a JSON reply;
2. validates the final text reply against the type;
3. on a mismatch, tells the model what was wrong and lets it try again,
   within ``max_iterations``.

The validated object lands in ``final_state.variables["output"]``. This works
with any model and provider; it needs no provider-specific JSON mode.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from miniagent.core.message import Message, MessageRole
from miniagent.core.validation import describe_validation_error

OutputT = TypeVar("OutputT", bound=BaseModel)

OUTPUT_VARIABLE = "output"
"""The ``State.variables`` key that holds the validated output object."""

# Models often wrap JSON in a Markdown code fence despite being asked not to.
_CODE_FENCE = re.compile(r"^```[A-Za-z]*\s*\n(?P<body>.*?)\n?```$", re.DOTALL)


def output_instructions(output_type: type[BaseModel]) -> str:
    """Return the system prompt addition that asks for ``output_type`` as JSON."""
    schema = json.dumps(output_type.model_json_schema(), indent=2)
    return (
        "When you give your final answer, reply with only a JSON object, with no "
        "other text and no code fence, matching this JSON Schema:\n"
        f"{schema}"
    )


def parse_output(text: str, output_type: type[OutputT]) -> OutputT:
    """Validate a final reply as ``output_type``, tolerating a code fence.

    Raises:
        ValidationError: If the reply is not JSON or does not match the type.
    """
    reply = text.strip()
    fenced = _CODE_FENCE.match(reply)
    if fenced:
        reply = fenced.group("body").strip()
    return output_type.model_validate_json(reply)


def correction_message(error: ValidationError) -> Message:
    """Build the message that tells the model why its reply was rejected."""
    return Message(
        role=MessageRole.USER,
        content=(
            "Your reply did not match the required JSON Schema: "
            f"{describe_validation_error(error, root_label='reply')}. "
            "Reply again with only the corrected JSON object."
        ),
    )
