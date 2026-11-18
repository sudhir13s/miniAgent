"""TextTransformTool — applies a text transformation (upper, lower, title, reverse, strip)."""

from __future__ import annotations

from typing import Any, ClassVar

from miniagent.core.exceptions import ToolInvocationError
from miniagent.tool.base import BaseTool, ToolSchema


class TextTransformTool(BaseTool):
    """Applies a text transformation (upper, lower, title, reverse, strip).

    Example::

        tool = TextTransformTool()
        result = await tool.invoke(text="hello world", transform="title")
        # "Hello World"
    """

    _TRANSFORMS: ClassVar[set[str]] = {"upper", "lower", "title", "reverse", "strip"}

    def __init__(self) -> None:
        """Initialise the text transform tool with its JSON schema."""
        super().__init__(
            ToolSchema(
                name="text_transform",
                description=(
                    "Transform text using one of: "
                    "'upper' (ALL CAPS), 'lower' (all lowercase), 'title' (Title Case), "
                    "'reverse' (esrever), 'strip' (remove leading/trailing whitespace)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to transform.",
                        },
                        "transform": {
                            "type": "string",
                            "enum": ["upper", "lower", "title", "reverse", "strip"],
                            "description": "The transformation to apply.",
                        },
                    },
                    "required": ["text", "transform"],
                },
            )
        )

    async def invoke(self, **kwargs: Any) -> str:
        """Apply the named ``transform`` to ``text`` and return the result."""
        text: str = kwargs.get("text", "")
        transform: str = kwargs.get("transform", "").lower()
        if transform not in self._TRANSFORMS:
            raise ToolInvocationError(
                f"Unknown transform '{transform}'. "
                f"Choose from: {', '.join(sorted(self._TRANSFORMS))}."
            )
        if transform == "upper":
            return text.upper()
        if transform == "lower":
            return text.lower()
        if transform == "title":
            return text.title()
        if transform == "reverse":
            return text[::-1]
        # strip
        return text.strip()
