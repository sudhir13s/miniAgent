"""Tests for the default behaviour BaseLLMProvider gives every provider."""

from __future__ import annotations

from collections.abc import AsyncIterator

from miniagent.core.message import Message, MessageRole, ToolCall
from miniagent.llm.base import BaseLLMProvider
from miniagent.llm.streaming import CompletionDone, StreamEvent, TextDelta
from miniagent.profile.model_config import ModelConfig
from miniagent.tool.base import ToolSchema

_MODEL = ModelConfig(id="custom/model")


class _MinimalProvider(BaseLLMProvider):
    """Implements only the abstract methods, so inherited defaults are tested."""

    async def complete(self, messages: list[Message], model_config: ModelConfig) -> Message:
        return Message(role=MessageRole.AI, content="reply")

    async def complete_with_tools(
        self,
        messages: list[Message],
        model_config: ModelConfig,
        tools: list[ToolSchema],
    ) -> tuple[Message, list[ToolCall]]:
        return Message(role=MessageRole.AI, content="reply"), []


class _ToolCallingProvider(_MinimalProvider):
    """Requests one tool call whenever tools are offered."""

    async def complete_with_tools(
        self,
        messages: list[Message],
        model_config: ModelConfig,
        tools: list[ToolSchema],
    ) -> tuple[Message, list[ToolCall]]:
        call = ToolCall(id="c1", tool_name="echo", arguments={"text": "x"})
        return Message(role=MessageRole.AI, content="[tool call]"), [call]


async def _collect(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in stream]


_ECHO_SCHEMA = ToolSchema(name="echo", description="Echo.", parameters={})


class TestFallbackStreamCompletion:
    """Providers without real streaming still stream: the whole reply in one piece."""

    async def test_should_yield_reply_text_then_the_finished_turn(self) -> None:
        events = await _collect(_MinimalProvider().stream_completion([], _MODEL))
        assert events[0] == TextDelta("reply")
        assert isinstance(events[1], CompletionDone)
        assert events[1].message.content == "reply"
        assert events[1].tool_calls == []

    async def test_should_use_tool_completion_when_tools_are_offered(self) -> None:
        events = await _collect(
            _ToolCallingProvider().stream_completion([], _MODEL, tools=[_ECHO_SCHEMA])
        )
        [done] = events
        assert isinstance(done, CompletionDone)
        assert [call.tool_name for call in done.tool_calls] == ["echo"]


class TestEstimatedTokenCount:
    def test_should_return_zero_for_no_messages(self) -> None:
        assert _MinimalProvider().count_tokens([], _MODEL) == 0

    def test_should_grow_with_content_length(self) -> None:
        provider = _MinimalProvider()
        short = [Message(role=MessageRole.USER, content="hi")]
        long = [Message(role=MessageRole.USER, content="hi " * 400)]
        assert provider.count_tokens(long, _MODEL) > provider.count_tokens(short, _MODEL)

    def test_should_be_roughly_four_characters_per_token(self) -> None:
        message = Message(role=MessageRole.USER, content="x" * 4000)
        estimate = _MinimalProvider().count_tokens([message], _MODEL)
        assert 1000 <= estimate <= 1010

    def test_should_count_requested_tool_calls(self) -> None:
        call = ToolCall(id="c1", tool_name="search", arguments={"query": "x" * 200})
        with_call = Message(role=MessageRole.AI, content="[tool call]", tool_calls=[call])
        without_call = Message(role=MessageRole.AI, content="[tool call]")
        provider = _MinimalProvider()
        assert provider.count_tokens([with_call], _MODEL) > provider.count_tokens(
            [without_call], _MODEL
        )
