"""TimestampTool — returns the current UTC date and time."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from miniagent.tool.base import BaseTool, ToolSchema


class TimestampTool(BaseTool):
    """Returns the current UTC timestamp in ISO 8601 format.

    Example::

        tool = TimestampTool()
        result = await tool.invoke()
        # "2025-06-08T14:32:00.123456+00:00"

        result = await tool.invoke(format="%Y-%m-%d %H:%M:%S UTC")
        # "2025-06-08 14:32:00 UTC"
    """

    def __init__(self) -> None:
        """Initialise the timestamp tool with its JSON schema."""
        super().__init__(
            ToolSchema(
                name="timestamp",
                description=(
                    "Return the current UTC date and time. "
                    "Pass an optional 'format' string (strftime syntax) to control output. "
                    "Default format is ISO 8601, e.g. '2025-06-08T14:32:00+00:00'."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "description": (
                                "Optional strftime format string. "
                                "E.g. '%Y-%m-%d' returns '2025-06-08'. "
                                "Omit for full ISO 8601 format."
                            ),
                        },
                    },
                    "required": [],
                },
            )
        )

    async def invoke(self, **kwargs: Any) -> str:
        """Return the current UTC time, formatted by the optional ``format`` arg."""
        fmt: str | None = kwargs.get("format")
        now = datetime.now(timezone.utc)
        if fmt:
            return now.strftime(fmt)
        return now.isoformat()
