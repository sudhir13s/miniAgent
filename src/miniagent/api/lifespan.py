"""Application lifespan: connect configured MCP servers for the life of the API.

Servers listed under ``mcp_servers:`` in the agents configuration are
connected once at startup and closed at shutdown; every request builds its
agent against these open toolsets instead of starting a server per request.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from miniagent.api.dependencies import resolve_config_path
from miniagent.core.exceptions import ConfigurationError
from miniagent.profile.mcp_server_config import MCPServerConfig
from miniagent.tool.toolset import BaseToolset, open_toolsets

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the configured toolsets on startup and close them on shutdown.

    Raises:
        ConfigurationError: If ``mcp_servers`` is invalid or needs the
            missing ``mcp`` extra, so a misconfigured API fails at startup.
    """
    async with open_toolsets(_configured_toolsets()) as toolsets:
        app.state.toolsets = toolsets
        logger.info("API started with %d MCP toolset(s)", len(toolsets))
        yield
    app.state.toolsets = []


def _configured_toolsets() -> list[BaseToolset]:
    """Build (unconnected) toolsets for the configuration's ``mcp_servers``."""
    config_path = resolve_config_path()
    if not config_path.exists():
        return []
    servers = MCPServerConfig.from_yaml(config_path)
    return _build_mcp_toolsets(servers) if servers else []


def _build_mcp_toolsets(servers: list[MCPServerConfig]) -> list[BaseToolset]:
    try:
        from miniagent.tool.mcp import MCPToolset
    except ImportError as exc:
        raise ConfigurationError(
            "The configuration defines mcp_servers, which need the optional 'mcp' extra: "
            "install miniagent[mcp].",
            cause=exc,
        ) from exc
    return [MCPToolset.from_config(server) for server in servers]
