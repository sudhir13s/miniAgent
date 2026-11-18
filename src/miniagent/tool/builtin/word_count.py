"""WordCountTool — counts words, characters, and lines in a text string."""

from __future__ import annotations

from typing import Any

from miniagent.tool.base import BaseTool, ToolSchema


class WordCountTool(BaseTool):
    """Counts words, characters, and lines in a text string.

    Example::

        tool = WordCountTool()
        result = await tool.invoke(text="Hello world! How are you?")
        # "words: 5, characters: 25, lines: 1"
    """

    def __init__(self) -> None:
        """Initialise the word count tool with its JSON schema."""
        super().__init__(
            ToolSchema(
                name="word_count",
                description=(
                    "Count the number of words, characters, and lines in a text. "
                    "Returns a summary string, e.g. 'words: 42, characters: 230, lines: 5'."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to analyze.",
                        },
                    },
                    "required": ["text"],
                },
            )
        )

    async def invoke(self, **kwargs: Any) -> str:
        """Count words, characters, and lines in ``text`` and return a summary."""
        text: str = kwargs.get("text", "")
        words = len(text.split())
        chars = len(text)
        lines = len(text.splitlines()) if text else 0
        return f"words: {words}, characters: {chars}, lines: {lines}"
