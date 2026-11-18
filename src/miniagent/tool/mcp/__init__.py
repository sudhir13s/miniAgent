"""MCP support: connect Model Context Protocol servers as agent toolsets.

Needs the optional ``mcp`` extra: install ``miniagent[mcp]``.

::

    from miniagent.tool.mcp import MCPToolset, StdioServerParameters

    github = MCPToolset.stdio(
        "github", StdioServerParameters(command="github-mcp-server", args=["stdio"])
    )
    async with github:
        agent = build_agent(profile, toolsets=[github])
"""

try:
    from mcp import StdioServerParameters
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "MCP support needs the optional 'mcp' extra: install miniagent[mcp]."
    ) from exc

from miniagent.tool.mcp.tool import MCPTool, MCPToolOptions
from miniagent.tool.mcp.toolset import MCPToolset

__all__ = ["MCPTool", "MCPToolOptions", "MCPToolset", "StdioServerParameters"]
