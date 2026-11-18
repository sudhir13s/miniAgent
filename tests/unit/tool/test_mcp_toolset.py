"""Tests for ``MCPToolset`` against a real MCP server running in-process.

The in-memory transport still goes through the MCP protocol layer, so tool
listing, argument passing and error reporting behave as over stdio or HTTP.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("mcp", reason="MCP support needs the 'mcp' extra")

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from miniagent.agent.agent import Agent
from miniagent.core.exceptions import ToolInvocationError
from miniagent.core.message import Message, MessageRole, ToolCall
from miniagent.core.state import State
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.model_config import ModelConfig
from miniagent.tool.base import BaseTool
from miniagent.tool.mcp import MCPToolOptions, MCPToolset, StdioServerParameters


def _calculator_server() -> MCPServer:
    server = MCPServer("calculator")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool()
    def divide(a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ToolError("Cannot divide by zero.")
        return a / b

    @server.tool()
    def broken() -> str:
        """Has a bug."""
        raise KeyError("secret_internal_detail")

    @server.tool(name="files.read")
    def read_file(path: str) -> str:
        """Read a file."""
        return f"contents of {path}"

    return server


def _toolset(options: MCPToolOptions | None = None) -> MCPToolset:
    return MCPToolset("calc", _calculator_server(), options)


async def _tools_by_name(toolset: MCPToolset) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in await toolset.list_tools()}


class TestListTools:
    async def test_tools_carry_the_servers_schema(self) -> None:
        async with _toolset() as toolset:
            tools = await _tools_by_name(toolset)

        assert set(tools) == {"add", "divide", "broken", "files_read"}
        assert tools["add"].schema.description == "Add two integers."
        assert set(tools["add"].schema.parameters["properties"]) == {"a", "b"}

    async def test_tool_filter_keeps_only_listed_tools(self) -> None:
        async with _toolset(MCPToolOptions(tool_filter={"add"})) as toolset:
            tools = await toolset.list_tools()

        assert [tool.name for tool in tools] == ["add"]

    async def test_prefix_is_added_to_tool_names(self) -> None:
        async with _toolset(MCPToolOptions(name_prefix="calc_")) as toolset:
            names = sorted(await _tools_by_name(toolset))

        assert names == ["calc_add", "calc_broken", "calc_divide", "calc_files_read"]

    async def test_timeout_is_applied_to_every_tool(self) -> None:
        async with _toolset(MCPToolOptions(timeout_seconds=5)) as toolset:
            tools = await toolset.list_tools()

        assert {tool.timeout_seconds for tool in tools} == {5}

    async def test_listing_before_connecting_explains_how_to_connect(self) -> None:
        with pytest.raises(ToolInvocationError, match="not connected"):
            await _toolset().list_tools()


class TestInvoke:
    async def test_returns_the_servers_text_result(self) -> None:
        async with _toolset() as toolset:
            tools = await _tools_by_name(toolset)
            assert await tools["add"].invoke(a=2, b=3) == "5"

    async def test_renamed_tool_still_calls_the_original_server_name(self) -> None:
        async with _toolset(MCPToolOptions(name_prefix="calc_")) as toolset:
            tools = await _tools_by_name(toolset)
            assert await tools["calc_files_read"].invoke(path="a.txt") == "contents of a.txt"

    async def test_server_tool_error_raises_tool_error_with_its_message(self) -> None:
        async with _toolset() as toolset:
            tools = await _tools_by_name(toolset)
            with pytest.raises(ToolInvocationError, match="Cannot divide by zero"):
                await tools["divide"].invoke(a=1, b=0)

    async def test_unexpected_server_error_still_raises_tool_error(self) -> None:
        """The server hides internal details; the model still learns the call failed."""
        async with _toolset() as toolset:
            tools = await _tools_by_name(toolset)
            with pytest.raises(ToolInvocationError, match="broken") as raised:
                await tools["broken"].invoke()

        assert "secret_internal_detail" not in str(raised.value)

    async def test_invoking_after_close_raises_tool_error(self) -> None:
        async with _toolset() as toolset:
            tools = await _tools_by_name(toolset)

        with pytest.raises(ToolInvocationError, match="not connected"):
            await tools["add"].invoke(a=1, b=2)


class TestLifecycle:
    async def test_connect_twice_keeps_one_session(self) -> None:
        toolset = _toolset()
        await toolset.connect()
        session = toolset.client
        await toolset.connect()

        assert toolset.client is session
        await toolset.close()

    async def test_can_reconnect_after_close(self) -> None:
        toolset = _toolset()
        async with toolset:
            pass
        async with toolset:
            assert "add" in await _tools_by_name(toolset)


class TestFactories:
    def test_stdio_builds_a_named_toolset(self) -> None:
        server = StdioServerParameters(command="github-mcp-server", args=["stdio"])
        assert MCPToolset.stdio("github", server).name == "github"

    def test_http_builds_a_named_toolset(self) -> None:
        toolset = MCPToolset.http("docs", "https://example.com/mcp", headers={"X-Key": "k"})
        assert toolset.name == "docs"


class TestWithAgent:
    async def test_agent_calls_an_mcp_tool_and_sees_its_result(self) -> None:
        tool_call = ToolCall(id="tc1", tool_name="add", arguments={"a": 20, "b": 22})
        provider = MagicMock()
        provider.complete_with_tools = AsyncMock(
            side_effect=[
                (Message(role=MessageRole.AI, content="[tool call]"), [tool_call]),
                (Message(role=MessageRole.AI, content="It is 42."), []),
            ]
        )
        profile = AgentProfile(name="mcp-agent", model=ModelConfig(id="openai/gpt-4o"))
        state = State(messages=[Message(role=MessageRole.USER, content="20 + 22?")])

        async with _toolset(MCPToolOptions(tool_filter={"add"})) as toolset:
            agent = Agent(profile, provider, toolsets=[toolset])
            final_state = await agent.run(state)

        tool_messages = [m for m in final_state.messages if m.role == MessageRole.TOOL]
        assert tool_messages[0].content == "42"
        assert tool_messages[0].is_error is False
