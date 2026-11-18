"""One MCP server tool exposed as a miniAgent tool, and the options that shape it."""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from mcp.types import TextContent

from miniagent.core.exceptions import ToolInvocationError
from miniagent.tool.base import (
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    BaseTool,
    ToolSchema,
    validate_tool_name,
)

if TYPE_CHECKING:
    from mcp import Client
    from mcp.types import CallToolResult, Tool

# MCP allows tool names such as "files.read"; model providers do not.
_INVALID_TOOL_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9_-]")


class MCPServerSession(Protocol):
    """What a tool needs from the toolset that owns its server session.

    Declared here so this module does not import the toolset, which imports
    this module to build its tools.
    """

    @property
    def name(self) -> str:
        """The server's name, used in descriptions and error messages."""
        ...

    @property
    def client(self) -> Client:
        """The connected MCP client."""
        ...


@dataclass(frozen=True)
class MCPToolOptions:
    """Which of a server's tools to expose, and how.

    Attributes:
        tool_filter: Server tool names to expose; ``None`` exposes all.
        name_prefix: Prepended to every tool name (e.g. ``"gh_"``) to keep
            tools from different servers apart.
        timeout_seconds: Per-call time limit for every tool; ``None`` disables it.
    """

    tool_filter: Collection[str] | None = None
    name_prefix: str = ""
    timeout_seconds: float | None = DEFAULT_TOOL_TIMEOUT_SECONDS

    def exposes(self, remote_name: str) -> bool:
        """Return whether the server tool named ``remote_name`` should be offered."""
        return self.tool_filter is None or remote_name in self.tool_filter


class MCPTool(BaseTool):
    """One tool on an MCP server; calls go through its owner's open session."""

    def __init__(self, remote: Tool, toolset: MCPServerSession, options: MCPToolOptions) -> None:
        """Adapt a server tool definition.

        Args:
            remote: The tool definition the server published.
            toolset: The owner of the server session, normally the
                ``MCPToolset``.
            options: Naming and timeout options.

        Raises:
            ValueError: If the provider-safe name is still invalid, e.g. too long.
        """
        local_name = options.name_prefix + _INVALID_TOOL_NAME_CHARACTERS.sub("_", remote.name)
        super().__init__(
            ToolSchema(
                name=validate_tool_name(local_name),
                description=remote.description
                or remote.title
                or f"Tool '{remote.name}' from MCP server '{toolset.name}'.",
                parameters=remote.input_schema,
            )
        )
        self.remote_name = remote.name
        self.timeout_seconds = options.timeout_seconds
        self._toolset = toolset

    async def invoke(self, **kwargs: Any) -> str:
        """Call the server-side tool with the model's arguments and return its text.

        Raises:
            ToolInvocationError: If the toolset is not connected or the server
                reports that the tool failed.
        """
        result = await self._toolset.client.call_tool(self.remote_name, kwargs)
        text = _result_text(result)
        if result.is_error:
            raise ToolInvocationError(text or f"MCP tool '{self.remote_name}' reported an error.")
        return text


def _result_text(result: CallToolResult) -> str:
    """Flatten a tool result into text: text blocks first, then structured data."""
    texts = [block.text for block in result.content if isinstance(block, TextContent)]
    if texts:
        return "\n".join(texts)
    if result.structured_content is not None:
        return json.dumps(result.structured_content, default=str)
    return ", ".join(f"[{block.type} content]" for block in result.content)
