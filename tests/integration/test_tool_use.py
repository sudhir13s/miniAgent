"""Integration tests for tool-use within a ReAct loop.

Uses a concrete EchoTool subclass, mocks LiteLLM to return a tool call on the
first turn and a text response on the second.  No live API calls made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from miniagent.core.message import MessageRole
from miniagent.core.state import State
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.model_config import ModelConfig
from miniagent.tool.base import BaseTool, ToolSchema
from tests.integration.conftest import make_agent

# ---------------------------------------------------------------------------
# Concrete tool implementations
# ---------------------------------------------------------------------------


class EchoTool(BaseTool):
    """Returns its ``text`` argument verbatim."""

    def __init__(self) -> None:
        super().__init__(
            ToolSchema(
                name="echo",
                description="Echoes the input text.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )
        self.call_count = 0

    async def invoke(self, **kwargs: Any) -> str:
        self.call_count += 1
        return kwargs.get("text", "")


class UpperCaseTool(BaseTool):
    """Uppercases the input text."""

    def __init__(self) -> None:
        super().__init__(
            ToolSchema(
                name="upper",
                description="Returns text in uppercase.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        )

    async def invoke(self, **kwargs: Any) -> str:
        return kwargs.get("text", "").upper()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_litellm_tool_call_response(
    tool_name: str, arguments: dict[str, Any], call_id: str = "tc-1"
) -> MagicMock:
    """Build a litellm response that requests a tool call."""
    func = MagicMock()
    func.name = tool_name
    import json

    func.arguments = json.dumps(arguments)

    tc = MagicMock()
    tc.id = call_id
    tc.function = func

    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [tc]

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_litellm_text_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture()
def agent_profile() -> AgentProfile:
    return AgentProfile(
        name="tool-agent",
        model=ModelConfig(id="openai/gpt-4o"),
        system_prompt="Use tools to answer questions.",
        max_iterations=5,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolUseIntegration:
    async def test_tool_invoked_once(
        self,
        base_state: State,
        mock_litellm: Any,
        agent_profile: AgentProfile,
    ) -> None:
        echo = EchoTool()
        agent = make_agent(agent_profile, tools={"echo": echo})

        mock_litellm.side_effect = [
            _make_litellm_tool_call_response("echo", {"text": "hello"}),
            _make_litellm_text_response("The echo said: hello"),
        ]

        await agent.run(base_state)

        assert echo.call_count == 1

    async def test_second_call_resends_assistant_tool_calls_before_tool_result(
        self,
        base_state: State,
        mock_litellm: Any,
        agent_profile: AgentProfile,
    ) -> None:
        """Providers reject a tool result whose assistant tool_call is missing."""
        agent = make_agent(agent_profile, tools={"echo": EchoTool()})
        mock_litellm.side_effect = [
            _make_litellm_tool_call_response("echo", {"text": "hi"}, call_id="call-42"),
            _make_litellm_text_response("Done."),
        ]

        await agent.run(base_state)

        sent = mock_litellm.call_args_list[1].kwargs["messages"]
        assistant_turn, tool_turn = sent[-2], sent[-1]
        assert assistant_turn["role"] == "assistant"
        assert assistant_turn["content"] is None
        assert assistant_turn["tool_calls"] == [
            {
                "id": "call-42",
                "type": "function",
                "function": {"name": "echo", "arguments": '{"text": "hi"}'},
            }
        ]
        assert tool_turn == {"role": "tool", "content": "hi", "tool_call_id": "call-42"}

    async def test_second_call_preserves_thinking_blocks(
        self,
        base_state: State,
        mock_litellm: Any,
        agent_profile: AgentProfile,
    ) -> None:
        """Claude with thinking enabled requires thinking blocks on tool-use turns."""
        thinking_blocks = [{"type": "thinking", "thinking": "Use echo.", "signature": "sig"}]
        tool_call_response = _make_litellm_tool_call_response("echo", {"text": "hi"})
        tool_call_response.choices[0].message.thinking_blocks = thinking_blocks
        tool_call_response.choices[0].message.reasoning_content = "Use echo."
        agent = make_agent(agent_profile, tools={"echo": EchoTool()})
        mock_litellm.side_effect = [tool_call_response, _make_litellm_text_response("Done.")]

        result = await agent.run(base_state)

        assistant_turn = mock_litellm.call_args_list[1].kwargs["messages"][-2]
        assert assistant_turn["thinking_blocks"] == thinking_blocks
        first_ai = next(m for m in result.messages if m.role == MessageRole.AI)
        assert first_ai.reasoning == "Use echo."
        assert first_ai.thinking_blocks == thinking_blocks

    async def test_tool_result_message_in_state(
        self,
        base_state: State,
        mock_litellm: Any,
        agent_profile: AgentProfile,
    ) -> None:
        echo = EchoTool()
        agent = make_agent(agent_profile, tools={"echo": echo})

        mock_litellm.side_effect = [
            _make_litellm_tool_call_response("echo", {"text": "ping"}),
            _make_litellm_text_response("Done."),
        ]

        result = await agent.run(base_state)

        tool_msgs = [m for m in result.messages if m.role == MessageRole.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "ping"

    async def test_final_ai_message_follows_tool_result(
        self,
        base_state: State,
        mock_litellm: Any,
        agent_profile: AgentProfile,
    ) -> None:
        echo = EchoTool()
        agent = make_agent(agent_profile, tools={"echo": echo})

        mock_litellm.side_effect = [
            _make_litellm_tool_call_response("echo", {"text": "world"}),
            _make_litellm_text_response("I echoed: world"),
        ]

        result = await agent.run(base_state)

        ai_msgs = [m for m in result.messages if m.role == MessageRole.AI]
        assert ai_msgs[-1].content == "I echoed: world"

    async def test_message_order_is_ai_tool_ai(
        self,
        base_state: State,
        mock_litellm: Any,
        agent_profile: AgentProfile,
    ) -> None:
        echo = EchoTool()
        agent = make_agent(agent_profile, tools={"echo": echo})

        mock_litellm.side_effect = [
            _make_litellm_tool_call_response("echo", {"text": "x"}),
            _make_litellm_text_response("Final answer."),
        ]

        result = await agent.run(base_state)

        new_msgs = [m for m in result.messages if m.role in (MessageRole.AI, MessageRole.TOOL)]
        roles = [m.role for m in new_msgs]
        assert roles == [MessageRole.AI, MessageRole.TOOL, MessageRole.AI]

    async def test_unknown_tool_gets_error_message_not_exception(
        self,
        base_state: State,
        mock_litellm: Any,
        agent_profile: AgentProfile,
    ) -> None:
        """LLM requests 'nonexistent_tool' — should get an error TOOL msg, not raise."""
        echo = EchoTool()
        agent = make_agent(agent_profile, tools={"echo": echo})

        mock_litellm.side_effect = [
            _make_litellm_tool_call_response("nonexistent_tool", {"x": 1}),
            _make_litellm_text_response("I couldn't use that tool."),
        ]

        result = await agent.run(base_state)

        tool_msgs = [m for m in result.messages if m.role == MessageRole.TOOL]
        assert len(tool_msgs) == 1
        assert "not registered" in tool_msgs[0].content

    async def test_two_different_tools_both_invoked(
        self,
        base_state: State,
        mock_litellm: Any,
        agent_profile: AgentProfile,
    ) -> None:
        echo = EchoTool()
        upper = UpperCaseTool()
        agent = make_agent(agent_profile, tools={"echo": echo, "upper": upper})

        # First call: two tool calls at once
        func_echo = MagicMock()
        func_echo.name = "echo"
        func_echo.arguments = '{"text": "hello"}'
        tc_echo = MagicMock()
        tc_echo.id = "tc-1"
        tc_echo.function = func_echo

        func_upper = MagicMock()
        func_upper.name = "upper"
        func_upper.arguments = '{"text": "world"}'
        tc_upper = MagicMock()
        tc_upper.id = "tc-2"
        tc_upper.function = func_upper

        msg1 = MagicMock()
        msg1.content = ""
        msg1.tool_calls = [tc_echo, tc_upper]
        choice1 = MagicMock()
        choice1.message = msg1
        resp1 = MagicMock()
        resp1.choices = [choice1]

        mock_litellm.side_effect = [
            resp1,
            _make_litellm_text_response("Used both tools."),
        ]

        result = await agent.run(base_state)

        tool_msgs = [m for m in result.messages if m.role == MessageRole.TOOL]
        assert len(tool_msgs) == 2
        contents = {m.content for m in tool_msgs}
        assert "hello" in contents
        assert "WORLD" in contents
