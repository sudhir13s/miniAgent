"""EchoTool — echoes the input message back unchanged."""

from __future__ import annotations

from typing import Any

from miniagent.tool.base import BaseTool, ToolSchema


class EchoTool(BaseTool):
    """Echoes the input message back unchanged.

    Useful for debugging agent tool-call flows and verifying that
    tools are being invoked with the correct arguments.

    Example::

        tool = EchoTool()
        result = await tool.invoke(message="hello")
        # "hello"
    """

    def __init__(self) -> None:
        """Initialise the echo tool with its JSON schema."""
        super().__init__(
            ToolSchema(
                name="echo",
                description=(
                    "Echo the input message back unchanged. "
                    "Useful for testing that tool calls are working correctly."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to echo back.",
                        },
                    },
                    "required": ["message"],
                },
            )
        )

    async def invoke(self, **kwargs: Any) -> str:
        """Return the ``message`` argument unchanged as a string."""
        return str(kwargs.get("message", ""))
