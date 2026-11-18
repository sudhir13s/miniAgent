"""JsonFormatterTool — parses and pretty-prints a JSON string."""

from __future__ import annotations

import json
from typing import Any

from miniagent.core.exceptions import ToolInvocationError
from miniagent.tool.base import BaseTool, ToolSchema


class JsonFormatterTool(BaseTool):
    r"""Parses and pretty-prints a JSON string.

    Example::

        tool = JsonFormatterTool()
        result = await tool.invoke(json_string='{"a":1,"b":true}')
        # '{\n  "a": 1,\n  "b": true\n}'
    """

    def __init__(self) -> None:
        """Initialise the JSON formatter tool with its JSON schema."""
        super().__init__(
            ToolSchema(
                name="json_formatter",
                description=(
                    "Parse and pretty-print a JSON string. "
                    "Returns the formatted JSON with indentation. "
                    "Raises an error if the input is not valid JSON."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "json_string": {
                            "type": "string",
                            "description": "The JSON string to format.",
                        },
                        "indent": {
                            "type": "integer",
                            "description": "Number of spaces for indentation (default: 2).",
                            "default": 2,
                        },
                    },
                    "required": ["json_string"],
                },
            )
        )

    async def invoke(self, **kwargs: Any) -> str:
        """Parse ``json_string`` and return it pretty-printed with ``indent``."""
        json_string: str = kwargs.get("json_string", "")
        indent: int = int(kwargs.get("indent", 2))
        try:
            parsed = json.loads(json_string)
            return json.dumps(parsed, indent=indent, ensure_ascii=False)
        except json.JSONDecodeError as exc:
            raise ToolInvocationError(f"Invalid JSON: {exc}", cause=exc) from exc
