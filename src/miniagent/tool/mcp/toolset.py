"""``MCPToolset``: the connection to one MCP server and its current tool list.

The server's tools are fetched at the start of every agent run, so tools the
server adds or removes are picked up without rebuilding the agent.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from miniagent.core.exceptions import ToolInvocationError
from miniagent.profile.mcp_server_config import MCPServerConfig
from miniagent.tool.base import BaseTool
from miniagent.tool.mcp.tool import MCPTool, MCPToolOptions
from miniagent.tool.toolset import BaseToolset

if TYPE_CHECKING:
    from mcp.client import Transport
    from mcp.server import MCPServer, Server

    # Everything ``mcp.Client`` can connect to.
    MCPTarget = Server[Any] | MCPServer | Transport | StdioServerParameters | str

logger = logging.getLogger(__name__)

# The MCP SDK's previous HTTP defaults: 30s to connect or send, 5 minutes to
# read a streamed response. httpx2's own 5-second default is too short.
_HTTP_TIMEOUT = httpx2.Timeout(30.0, read=300.0)


class MCPToolset(BaseToolset):
    """Tools published by an MCP server.

    Build one with ``MCPToolset.stdio`` (a local subprocess) or
    ``MCPToolset.http`` (a Streamable HTTP endpoint), then use it as an async
    context manager so the connection is opened and always closed.
    """

    def __init__(self, name: str, target: MCPTarget, options: MCPToolOptions | None = None) -> None:
        """Create a toolset for any target ``mcp.Client`` accepts.

        Args:
            name: Toolset identifier that profiles select with ``toolsets: [name]``.
            target: A URL, ``StdioServerParameters``, transport, or in-process
                server object (useful in tests).
            options: Which tools to expose and how; defaults expose all tools.
        """
        super().__init__(name)
        self._target = target
        self._options = options or MCPToolOptions()
        self._http_headers: Mapping[str, str] | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._client: Client | None = None

    @classmethod
    def stdio(
        cls,
        name: str,
        server: StdioServerParameters,
        options: MCPToolOptions | None = None,
    ) -> MCPToolset:
        """Launch an MCP server as a subprocess and talk to it over stdin/stdout.

        Args:
            name: Toolset identifier.
            server: Command, arguments, environment and working directory, e.g.
                ``StdioServerParameters(command="uvx", args=["mcp-server-git"])``.
                Without ``env``, the MCP SDK passes only a minimal safe
                environment, not your full shell.
            options: Which tools to expose and how.

        Returns:
            An unconnected toolset.
        """
        return cls(name, server, options)

    @classmethod
    def http(
        cls,
        name: str,
        url: str,
        options: MCPToolOptions | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> MCPToolset:
        """Connect to an MCP server's Streamable HTTP endpoint.

        For auth flows, proxies or custom TLS, build
        ``streamable_http_client(url, http_client=...)`` yourself and pass it
        to ``MCPToolset(name, transport)``.

        Args:
            name: Toolset identifier.
            url: The server's MCP endpoint, e.g. ``"https://example.com/mcp"``.
            options: Which tools to expose and how.
            headers: Extra request headers, e.g. ``{"Authorization": "Bearer ..."}``.

        Returns:
            An unconnected toolset.
        """
        toolset = cls(name, url, options)
        toolset._http_headers = headers
        return toolset

    @classmethod
    def from_config(cls, config: MCPServerConfig) -> MCPToolset:
        """Build a toolset from an ``mcp_servers:`` entry of agents.yaml.

        ``${VAR}`` references in ``env`` and ``headers`` are resolved now,
        from the process environment.

        Args:
            config: The validated server definition.

        Returns:
            An unconnected toolset.

        Raises:
            ConfigurationError: If a referenced environment variable is not set.
        """
        options = MCPToolOptions(
            tool_filter=config.tool_filter,
            name_prefix=config.name_prefix,
            timeout_seconds=config.timeout_seconds,
        )
        if config.command is not None:
            server = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.resolved_env() or None,
                cwd=config.cwd,
            )
            return cls.stdio(config.name, server, options)
        # MCPServerConfig guarantees a url whenever there is no command.
        url = config.url or ""
        return cls.http(config.name, url, options, headers=config.resolved_headers() or None)

    @property
    def client(self) -> Client:
        """The open MCP session.

        Raises:
            ToolInvocationError: If the toolset is not connected.
        """
        if self._client is None:
            raise ToolInvocationError(
                f"MCP toolset '{self.name}' is not connected; "
                f"use it as 'async with toolset:' or call connect() first."
            )
        return self._client

    async def connect(self) -> None:
        """Start or reach the server and complete the MCP handshake."""
        if self._client is not None:
            return
        exit_stack = AsyncExitStack()
        try:
            target = await self._open_target(exit_stack)
            self._client = await exit_stack.enter_async_context(Client(target))
        except BaseException:
            await exit_stack.aclose()
            raise
        self._exit_stack = exit_stack
        logger.info("Connected to MCP server for toolset '%s'", self.name)

    async def close(self) -> None:
        """Close the session, and the subprocess or HTTP client it owns."""
        exit_stack, self._exit_stack, self._client = self._exit_stack, None, None
        if exit_stack is not None:
            await exit_stack.aclose()

    async def list_tools(self) -> list[BaseTool]:
        """Return the server's current tools, following pagination.

        Raises:
            ToolInvocationError: If the toolset is not connected.
        """
        tools: list[BaseTool] = []
        cursor: str | None = None
        while True:
            page = await self.client.list_tools(cursor=cursor)
            tools.extend(
                MCPTool(remote, self, self._options)
                for remote in page.tools
                if self._options.exposes(remote.name)
            )
            cursor = page.next_cursor
            if not cursor:
                return tools

    async def _open_target(self, exit_stack: AsyncExitStack) -> MCPTarget:
        """Return what ``Client`` connects to, creating an HTTP client for headers."""
        # Headers are only ever set by ``http``, whose target is the URL.
        if self._http_headers is None or not isinstance(self._target, str):
            return self._target
        http_client = await exit_stack.enter_async_context(
            httpx2.AsyncClient(headers=dict(self._http_headers), timeout=_HTTP_TIMEOUT)
        )
        return streamable_http_client(self._target, http_client=http_client)
