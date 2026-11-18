"""Integration tests: real MCP server subprocesses, from code, from YAML, and via the API."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("mcp", reason="MCP support needs the 'mcp' extra")

from fastapi.testclient import TestClient

from miniagent.agent import build_agent
from miniagent.api.app import create_app
from miniagent.api.dependencies import reset_profile_cache
from miniagent.core.message import Message, MessageRole, ToolCall
from miniagent.core.state import State
from miniagent.profile import AgentProfile, MCPServerConfig
from miniagent.tool.mcp import MCPToolset, StdioServerParameters
from miniagent.tool.toolset import open_toolsets

_SERVER_SCRIPT = Path(__file__).with_name("mcp_echo_server.py")


def _write_config(tmp_path: Path) -> Path:
    """Write an agents.yaml with one stdio MCP server and an agent that uses it."""
    config = {
        "mcp_servers": [
            {"name": "echo", "command": sys.executable, "args": [str(_SERVER_SCRIPT)]},
        ],
        "agents": [
            {"name": "shouter", "model": {"id": "openai/gpt-4o"}, "toolsets": ["echo"]},
        ],
    }
    config_file = tmp_path / "agents.yaml"
    config_file.write_text(json.dumps(config), encoding="utf-8")  # JSON is valid YAML
    return config_file


def _litellm_response(*, content: str | None, tool_call: ToolCall | None = None) -> MagicMock:
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    message.thinking_blocks = None
    message.reasoning_content = None
    if tool_call is not None:
        function = MagicMock()
        function.name = tool_call.tool_name
        function.arguments = json.dumps(tool_call.arguments)
        litellm_call = MagicMock()
        litellm_call.id = tool_call.id
        litellm_call.function = function
        message.tool_calls = [litellm_call]
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


async def test_stdio_server_lists_and_runs_its_tool() -> None:
    server = StdioServerParameters(command=sys.executable, args=[str(_SERVER_SCRIPT)])

    async with MCPToolset.stdio("echo", server) as toolset:
        [shout] = await toolset.list_tools()
        result = await shout.invoke(text="hello")

    assert shout.name == "shout"
    assert result == "HELLO"


async def test_yaml_configured_server_serves_an_agent(tmp_path: Path) -> None:
    config_file = _write_config(tmp_path)
    [profile] = AgentProfile.from_yaml(config_file)
    toolsets = [MCPToolset.from_config(server) for server in MCPServerConfig.from_yaml(config_file)]
    tool_call = ToolCall(id="tc1", tool_name="shout", arguments={"text": "hi"})
    provider = MagicMock()
    provider.complete_with_tools = AsyncMock(
        side_effect=[
            (Message(role=MessageRole.AI, content="[tool call]"), [tool_call]),
            (Message(role=MessageRole.AI, content="Done."), []),
        ]
    )

    async with open_toolsets(toolsets):
        agent = build_agent(profile, toolsets=toolsets, llm_provider=provider)
        final_state = await agent.run(State(messages=[Message(role=MessageRole.USER, content="?")]))

    [tool_message] = [m for m in final_state.messages if m.role == MessageRole.TOOL]
    assert tool_message.content == "HI"


def test_api_connects_configured_servers_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTS_CONFIG_PATH", str(_write_config(tmp_path)))
    reset_profile_cache()
    tool_call = ToolCall(id="tc1", tool_name="shout", arguments={"text": "api"})
    completions = AsyncMock(
        side_effect=[
            _litellm_response(content=None, tool_call=tool_call),
            _litellm_response(content="It said API."),
        ]
    )

    with (
        patch("miniagent.llm.litellm_provider.litellm.acompletion", completions),
        TestClient(create_app()) as client,
    ):
        response = client.post("/agents/shouter/run", json={"message": "Shout api"})

    reset_profile_cache()
    assert response.status_code == 200, response.text
    assert response.json()["response"] == "It said API."
    second_call_messages: list[dict[str, Any]] = completions.call_args_list[1].kwargs["messages"]
    assert second_call_messages[-1] == {"role": "tool", "content": "API", "tool_call_id": "tc1"}
