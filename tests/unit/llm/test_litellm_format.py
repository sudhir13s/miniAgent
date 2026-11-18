"""Unit tests for converting messages to and from LiteLLM's OpenAI format."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from miniagent.core.exceptions import LLMProviderError
from miniagent.core.message import Message, MessageRole, ToolCall
from miniagent.llm.litellm_format import (
    TOOL_CALL_PLACEHOLDER,
    to_ai_turn,
    to_litellm_messages,
    to_litellm_tools,
)
from miniagent.tool.base import ToolSchema


def _reply(content: str | None, tool_calls: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _raw_call(call_id: str, name: str | None, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class TestToLiteLLMMessages:
    def test_should_map_roles_to_openai_names(self) -> None:
        messages = [
            Message(role=MessageRole.SYSTEM, content="rules"),
            Message(role=MessageRole.USER, content="hi"),
            Message(role=MessageRole.AI, content="hello"),
        ]
        assert [entry["role"] for entry in to_litellm_messages(messages)] == [
            "system",
            "user",
            "assistant",
        ]

    def test_should_send_tool_requests_back_with_empty_content(self) -> None:
        call = ToolCall(id="c1", tool_name="calc", arguments={"x": 1})
        message = Message(role=MessageRole.AI, content=TOOL_CALL_PLACEHOLDER, tool_calls=[call])
        [entry] = to_litellm_messages([message])
        assert entry["content"] is None
        assert entry["tool_calls"] == [
            {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": '{"x": 1}'}}
        ]

    def test_should_link_tool_results_to_their_call(self) -> None:
        result = Message(role=MessageRole.TOOL, content="2", tool_call_id="c1")
        assert to_litellm_messages([result]) == [
            {"role": "tool", "content": "2", "tool_call_id": "c1"}
        ]


class TestToLiteLLMTools:
    def test_should_wrap_schemas_as_functions(self) -> None:
        schema = ToolSchema(name="calc", description="Math.", parameters={"type": "object"})
        assert to_litellm_tools([schema]) == [
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "Math.",
                    "parameters": {"type": "object"},
                },
            }
        ]


class TestToAITurn:
    def test_should_keep_text_replies_as_they_are(self) -> None:
        message, tool_calls = to_ai_turn(_reply("Hello."), "openai/gpt-4o")
        assert message.content == "Hello."
        assert message.model == "openai/gpt-4o"
        assert (message.reasoning, message.thinking_blocks, message.metadata) == (None, [], {})
        assert tool_calls == []

    def test_should_keep_reasoning_and_thinking_blocks(self) -> None:
        reply = _reply("Done.")
        reply.reasoning_content = "First add, then check."
        reply.thinking_blocks = [{"type": "thinking", "signature": "sig"}]

        message, _ = to_ai_turn(reply, "anthropic/claude-sonnet-4-6")

        assert message.reasoning == "First add, then check."
        assert message.thinking_blocks == [{"type": "thinking", "signature": "sig"}]
        [entry] = to_litellm_messages([message])
        assert entry["thinking_blocks"] == [{"type": "thinking", "signature": "sig"}]

    def test_should_store_tool_calls_and_use_placeholder_content(self) -> None:
        reply = _reply(None, [_raw_call("c1", "calc", '{"x": 1}')])
        message, tool_calls = to_ai_turn(reply, "openai/gpt-4o")
        assert tool_calls == [ToolCall(id="c1", tool_name="calc", arguments={"x": 1})]
        assert message.content == TOOL_CALL_PLACEHOLDER
        assert message.tool_calls == tool_calls

    def test_should_reject_a_reply_with_no_text_and_no_tool_calls(self) -> None:
        with pytest.raises(LLMProviderError, match="empty reply"):
            to_ai_turn(_reply(None), "openai/gpt-4o")

    def test_should_turn_malformed_arguments_into_empty_arguments(self) -> None:
        _, [call] = to_ai_turn(_reply(None, [_raw_call("c1", "calc", "{not json")]), "m")
        assert call.arguments == {}

    def test_should_reject_a_tool_call_without_a_tool_name(self) -> None:
        with pytest.raises(LLMProviderError, match="without naming the tool"):
            to_ai_turn(_reply(None, [_raw_call("c1", None, "{}")]), "m")

    def test_should_round_trip_through_the_request_format(self) -> None:
        message, _ = to_ai_turn(_reply(None, [_raw_call("c1", "calc", '{"x": 1}')]), "m")
        [entry] = to_litellm_messages([message])
        assert entry["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
