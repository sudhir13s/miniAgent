"""FileReaderTool — reads a local file and returns its text content."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniagent.core.exceptions import ToolInvocationError
from miniagent.tool.base import BaseTool, ToolSchema

DEFAULT_MAX_CHARS = 10000


class FileReaderTool(BaseTool):
    r"""Reads a local file and returns its text content.

    Paths are resolved relative to the current working directory.
    Access is restricted to paths within the working directory tree
    to prevent directory traversal.

    Example::

        tool = FileReaderTool()
        result = await tool.invoke(path="README.md")
        # "# miniAgent\n..."
    """

    def __init__(self) -> None:
        """Initialise the file reader tool with its JSON schema."""
        super().__init__(
            ToolSchema(
                name="file_reader",
                description=(
                    "Read a local file and return its text content. "
                    "The path is relative to the current working directory. "
                    "Only files within the working directory are accessible."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Relative path to the file, e.g. 'README.md' or 'data/input.txt'."
                            ),
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": (
                                f"Maximum characters to return (default: {DEFAULT_MAX_CHARS})."
                            ),
                            "default": DEFAULT_MAX_CHARS,
                        },
                    },
                    "required": ["path"],
                },
            )
        )

    async def invoke(self, **kwargs: Any) -> str:
        """Read the file at ``path`` (within cwd) and return up to ``max_chars``."""
        rel_path: str = kwargs.get("path", "")
        max_chars: int = int(kwargs.get("max_chars", DEFAULT_MAX_CHARS))
        if not rel_path:
            raise ToolInvocationError("Path cannot be empty.")

        try:
            cwd = Path.cwd().resolve()
            target = (cwd / rel_path).resolve()

            # Restrict to working directory subtree
            target.relative_to(cwd)
        except ValueError as exc:
            raise ToolInvocationError(
                f"Access denied: '{rel_path}' is outside the working directory."
            ) from exc

        if not target.exists():
            raise ToolInvocationError(f"File not found: '{rel_path}'.")
        if not target.is_file():
            raise ToolInvocationError(f"'{rel_path}' is not a file.")

        try:
            content = target.read_text(encoding="utf-8")
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n... [truncated at {max_chars} characters]"
            return content
        except Exception as exc:
            raise ToolInvocationError(f"Failed to read '{rel_path}': {exc}", cause=exc) from exc
