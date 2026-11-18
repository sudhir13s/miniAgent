"""Abstract base class defining the contract for all miniAgent tools.

Tools are callable units that agents invoke to perform actions beyond
text generation, such as web searches, API calls, or computations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from miniagent.core.naming import validate_identifier

DEFAULT_TOOL_TIMEOUT_SECONDS = 60.0
# Longest function name OpenAI, Anthropic and Gemini all accept.
MAX_TOOL_NAME_LENGTH = 64


def validate_tool_name(name: str) -> str:
    """Ensure a tool name is one every major model provider accepts.

    Args:
        name: The proposed tool name.

    Returns:
        The validated name.

    Raises:
        ValueError: If the name has characters other than letters, digits,
            hyphens and underscores, or is longer than ``MAX_TOOL_NAME_LENGTH``.
    """
    validate_identifier(name, "Tool")
    if len(name) > MAX_TOOL_NAME_LENGTH:
        raise ValueError(
            f"Tool name '{name}' is {len(name)} characters; the limit is {MAX_TOOL_NAME_LENGTH}."
        )
    return name


class ToolSchema(BaseModel):
    """Schema describing a tool's interface for LLM function calling.

    Attributes:
        name: Unique name of the tool.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema-compatible dictionary describing
            the tool's expected input parameters.
    """

    name: str = Field(
        ...,
        min_length=1,
        description="Unique name of the tool.",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of what the tool does.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing expected input parameters.",
    )

    model_config = {"frozen": True}


class BaseTool(ABC):
    """Abstract base class for all tools in the miniAgent framework.

    Each tool exposes a schema for LLM function calling and an
    ``invoke`` method that performs the actual work.

    Attributes:
        schema: The tool's interface description for the LLM.
        timeout_seconds: Maximum time one ``invoke`` call may take before the
            agent abandons it and reports a timeout to the model. Subclasses
            override it; ``None`` disables the limit.
    """

    timeout_seconds: float | None = DEFAULT_TOOL_TIMEOUT_SECONDS

    def __init__(self, schema: ToolSchema) -> None:
        """Initialize the tool with its schema.

        Args:
            schema: The tool's interface description.
        """
        self.schema = schema

    @property
    def name(self) -> str:
        """Return the tool's unique name.

        Returns:
            The tool name from its schema.
        """
        return self.schema.name

    @abstractmethod
    async def invoke(self, **kwargs: Any) -> str:
        """Run the tool with the provided arguments.

        Args:
            **kwargs: Keyword arguments matching the tool's parameter
                schema.

        Returns:
            A string representation of the tool's output.

        Raises:
            ToolInvocationError: If the tool cannot complete the request.
                The agent reports any exception raised here to the model as
                an error result, so the message should help the model retry.
        """
