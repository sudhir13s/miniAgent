"""UuidTool — generates a UUID (version 1 or 4)."""

from __future__ import annotations

import uuid
from typing import Any

from miniagent.core.exceptions import ToolInvocationError
from miniagent.tool.base import BaseTool, ToolSchema


class UuidTool(BaseTool):
    """Generates a UUID (version 1 or 4).

    Example::

        tool = UuidTool()
        result = await tool.invoke()
        # "550e8400-e29b-41d4-a716-446655440000"
    """

    def __init__(self) -> None:
        """Initialise the UUID tool with its JSON schema."""
        super().__init__(
            ToolSchema(
                name="uuid_generator",
                description=(
                    "Generate a universally unique identifier (UUID). "
                    "Version 4 (random) is the default. Version 1 uses the host MAC address."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "version": {
                            "type": "integer",
                            "enum": [1, 4],
                            "description": "UUID version: 1 (time-based) or 4 (random, default).",
                            "default": 4,
                        },
                    },
                    "required": [],
                },
            )
        )

    async def invoke(self, **kwargs: Any) -> str:
        """Generate and return a UUID of the requested ``version`` (1 or 4)."""
        version: int = int(kwargs.get("version", 4))
        if version == 1:
            return str(uuid.uuid1())
        if version == 4:
            return str(uuid.uuid4())
        raise ToolInvocationError(f"Unsupported UUID version: {version}. Use 1 or 4.")
