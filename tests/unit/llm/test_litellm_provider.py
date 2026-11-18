"""Tests for LiteLLMProvider and provider registry.

All tests mock ``litellm.acompletion`` so no real API calls are made.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)

from miniagent.core.exceptions import LLMProviderError
from miniagent.core.message import Message, MessageRole, ToolCall
from miniagent.llm.litellm_format import to_litellm_messages, to_litellm_parameters
from miniagent.llm.litellm_provider import LiteLLMProvider
from miniagent.llm.registry import (
    get_default_provider,
    reset_default_provider,
    set_default_provider,
)
from miniagent.llm.streaming import CompletionDone, StreamEvent, TextDelta
from miniagent.profile.model_config import ModelConfig
from miniagent.tool.base import ToolSchema

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_response(content: str, tool_calls: list[Any] | None = None) -> MagicMock:
    """Build a mock litellm response object."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_tool_call_obj(tc_id: str, name: str, arguments: str) -> MagicMock:
    """Build a mock litellm tool_call object."""
    func = MagicMock()
    func.name = name
    func.arguments = arguments
    tc = MagicMock()
    tc.id = tc_id
    tc.function = func
    return tc


@pytest.fixture()
def provider() -> LiteLLMProvider:
    return LiteLLMProvider()


@pytest.fixture()
def model_config() -> ModelConfig:
    return ModelConfig(id="openai/gpt-4o")


@pytest.fixture()
def user_messages() -> list[Message]:
    return [Message(role=MessageRole.USER, content="Hello")]


@pytest.fixture()
def mock_litellm_complete() -> Any:
    """Patch litellm.acompletion for complete() and complete_with_tools()."""
    target = "miniagent.llm.litellm_provider.litellm.acompletion"
    with patch(target, new_callable=AsyncMock) as mock_fn:
        # Wrap in a namespace-style object so tests can do mock_litellm_complete.acompletion
        class _Ns:
            acompletion = mock_fn

        yield _Ns()


# ---------------------------------------------------------------------------
# complete() tests
# ---------------------------------------------------------------------------


class TestComplete:
    """LiteLLMProvider.complete()."""

    async def test_returns_ai_message(
        self,
        provider: LiteLLMProvider,
        model_config: ModelConfig,
        user_messages: list[Message],
        mock_litellm_complete: Any,
    ) -> None:
        mock_litellm_complete.acompletion.return_value = _make_response("Hello back!")
        result = await provider.complete(user_messages, model_config)
        assert result.role == MessageRole.AI
        assert result.content == "Hello back!"

    async def test_records_which_model_replied(
        self,
        provider: LiteLLMProvider,
        model_config: ModelConfig,
        user_messages: list[Message],
        mock_litellm_complete: Any,
    ) -> None:
        mock_litellm_complete.acompletion.return_value = _make_response("ok")
        result = await provider.complete(user_messages, model_config)
        assert result.model == "openai/gpt-4o"

    async def test_passes_model_string_to_litellm(
        self,
        provider: LiteLLMProvider,
        model_config: ModelConfig,
        user_messages: list[Message],
        mock_litellm_complete: Any,
    ) -> None:
        mock_litellm_complete.acompletion.return_value = _make_response("ok")
        await provider.complete(user_messages, model_config)
        call_kwargs = mock_litellm_complete.acompletion.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o"

    async def test_passes_temperature_to_litellm(
        self,
        provider: LiteLLMProvider,
        user_messages: list[Message],
        mock_litellm_complete: Any,
    ) -> None:
        cfg = ModelConfig(id="openai/gpt-4o", temperature=0.3)
        mock_litellm_complete.acompletion.return_value = _make_response("ok")
        await provider.complete(user_messages, cfg)
        call_kwargs = mock_litellm_complete.acompletion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3

    async def test_extra_params_passed_through(
        self,
        provider: LiteLLMProvider,
        user_messages: list[Message],
        mock_litellm_complete: Any,
    ) -> None:
        cfg = ModelConfig(
            id="openai/gpt-4o",
            extra_params={"stop": ["DONE"]},
        )
        mock_litellm_complete.acompletion.return_value = _make_response("ok")
        await provider.complete(user_messages, cfg)
        call_kwargs = mock_litellm_complete.acompletion.call_args.kwargs
        assert call_kwargs["stop"] == ["DONE"]

    async def test_raises_llm_provider_error_on_exception(
        self,
        provider: LiteLLMProvider,
        model_config: ModelConfig,
        user_messages: list[Message],
        mock_litellm_complete: Any,
    ) -> None:
        mock_litellm_complete.acompletion.side_effect = RuntimeError("API error")
        with pytest.raises(LLMProviderError) as exc_info:
            await provider.complete(user_messages, model_config)
        assert "openai/gpt-4o" in str(exc_info.value)

    async def test_raises_when_the_model_returns_nothing(
        self,
        provider: LiteLLMProvider,
        model_config: ModelConfig,
        user_messages: list[Message],
        mock_litellm_complete: Any,
    ) -> None:
        mock_litellm_complete.acompletion.return_value = _make_response("")
        with pytest.raises(LLMProviderError, match="empty reply"):
            await provider.complete(user_messages, model_config)

    async def test_system_message_converted_to_system_role(
        self,
        provider: LiteLLMProvider,
        model_config: ModelConfig,
        mock_litellm_complete: Any,
    ) -> None:
        messages = [Message(role=MessageRole.SYSTEM, content="You are helpful.")]
        mock_litellm_complete.acompletion.return_value = _make_response("ok")
        await provider.complete(messages, model_config)
        sent = mock_litellm_complete.acompletion.call_args.kwargs["messages"]
        assert sent[0]["role"] == "system"


# ---------------------------------------------------------------------------
# complete_with_tools() tests
# ---------------------------------------------------------------------------


class TestCompleteWithTools:
    """LiteLLMProvider.complete_with_tools()."""

    def _schema(self) -> ToolSchema:
        return ToolSchema(
            name="calculator",
            description="Do math",
            parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
        )

    async def test_returns_message_and_empty_tool_calls_when_no_tools_invoked(
        self,
        provider: LiteLLMProvider,
        model_config: ModelConfig,
        user_messages: list[Message],
        mock_litellm_complete: Any,
    ) -> None:
        mock_litellm_complete.acompletion.return_value = _make_response(
            "The answer is 4.", tool_calls=None
        )
        msg, tool_calls = await provider.complete_with_tools(
            user_messages, model_config, [self._schema()]
        )
        assert msg.role == MessageRole.AI
        assert msg.content == "The answer is 4."
        assert tool_calls == []

    async def test_parses_tool_calls_from_response(
        self,
        provider: LiteLLMProvider,
        model_config: ModelConfig,
        user_messages: list[Message],
        mock_litellm_complete: Any,
    ) -> None:
        tc_obj = _make_tool_call_obj("call_1", "calculator", '{"expr": "2+2"}')
        mock_litellm_complete.acompletion.return_value = _make_response("", tool_calls=[tc_obj])
        _msg, tool_calls = await provider.complete_with_tools(
            user_messages, model_config, [self._schema()]
        )
        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0], ToolCall)
        assert tool_calls[0].tool_name == "calculator"
        assert tool_calls[0].arguments == {"expr": "2+2"}
        assert tool_calls[0].id == "call_1"

    async def test_message_content_is_placeholder_when_only_tool_calls(
        self,
        provider: LiteLLMProvider,
        model_config: ModelConfig,
        user_messages: list[Message],
        mock_litellm_complete: Any,
    ) -> None:
        tc_obj = _make_tool_call_obj("call_1", "calculator", '{"expr": "3*3"}')
        mock_litellm_complete.acompletion.return_value = _make_response("", tool_calls=[tc_obj])
        msg, _ = await provider.complete_with_tools(user_messages, model_config, [self._schema()])
        assert msg.content == "[tool call]"

    async def test_raises_on_provider_error(
        self,
        provider: LiteLLMProvider,
        model_config: ModelConfig,
        user_messages: list[Message],
        mock_litellm_complete: Any,
    ) -> None:
        mock_litellm_complete.acompletion.side_effect = RuntimeError("timeout")
        with pytest.raises(LLMProviderError):
            await provider.complete_with_tools(user_messages, model_config, [self._schema()])

    async def test_sets_tool_choice_auto(
        self,
        provider: LiteLLMProvider,
        model_config: ModelConfig,
        user_messages: list[Message],
        mock_litellm_complete: Any,
    ) -> None:
        mock_litellm_complete.acompletion.return_value = _make_response("ok")
        await provider.complete_with_tools(user_messages, model_config, [self._schema()])
        call_kwargs = mock_litellm_complete.acompletion.call_args.kwargs
        assert call_kwargs["tool_choice"] == "auto"

    async def test_handles_malformed_json_arguments_gracefully(
        self,
        provider: LiteLLMProvider,
        model_config: ModelConfig,
        user_messages: list[Message],
        mock_litellm_complete: Any,
    ) -> None:
        tc_obj = _make_tool_call_obj("call_1", "calculator", "not-json{{{")
        mock_litellm_complete.acompletion.return_value = _make_response("", tool_calls=[tc_obj])
        _msg, tool_calls = await provider.complete_with_tools(
            user_messages, model_config, [self._schema()]
        )
        assert tool_calls[0].arguments == {}


# ---------------------------------------------------------------------------
# to_litellm_parameters() tests
# ---------------------------------------------------------------------------


class TestToLiteLLMParameters:
    """Model settings become ``litellm.acompletion`` keyword arguments."""

    def test_standard_fields(self) -> None:
        cfg = ModelConfig(id="openai/gpt-4o", temperature=0.5, max_tokens=2048)
        assert to_litellm_parameters(cfg) == {
            "model": "openai/gpt-4o",
            "temperature": 0.5,
            "max_tokens": 2048,
            "top_p": 1.0,
        }

    def test_top_k_included_only_when_set(self) -> None:
        assert to_litellm_parameters(ModelConfig(id="gemini/gemini-2.0-flash", top_k=40))[
            "top_k"
        ] == (40)
        assert "top_k" not in to_litellm_parameters(ModelConfig(id="openai/gpt-4o"))

    def test_extra_params_are_added_and_can_override(self) -> None:
        cfg = ModelConfig(
            id="openai/gpt-4o",
            temperature=0.2,
            extra_params={"stop": ["END"], "temperature": 0.9},
        )
        parameters = to_litellm_parameters(cfg)
        assert parameters["stop"] == ["END"]
        assert parameters["temperature"] == 0.9


# ---------------------------------------------------------------------------
# stream_completion() tests
# ---------------------------------------------------------------------------


def _stream_chunk(delta: Delta, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-1",
        model="gpt-4o",
        choices=[StreamingChoices(index=0, delta=delta, finish_reason=finish_reason)],
    )


def _tool_call_fragment(arguments: str, *, first: bool) -> Delta:
    """One piece of a streamed tool call; only the first piece names it."""
    return Delta(
        tool_calls=[
            ChatCompletionDeltaToolCall(
                index=0,
                id="call_1" if first else None,
                type="function" if first else None,
                function=Function(name="calculator" if first else None, arguments=arguments),
            )
        ]
    )


async def _replay(chunks: list[ModelResponseStream]) -> AsyncIterator[ModelResponseStream]:
    for chunk in chunks:
        yield chunk


async def _collect(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in stream]


class TestStreamCompletion:
    """stream_completion() yields text as it arrives, then the assembled turn."""

    async def test_should_stream_text_then_finish_with_the_full_message(
        self, provider: LiteLLMProvider, model_config: ModelConfig, user_messages: list[Message]
    ) -> None:
        streaming_config = model_config.model_copy(
            update={"extra_params": {"mock_response": "Hello there, friend."}}
        )
        events = await _collect(provider.stream_completion(user_messages, streaming_config))

        *deltas, done = events
        assert len(deltas) > 1, "LiteLLM's mock streams the reply in several chunks"
        assert "".join(delta.text for delta in deltas if isinstance(delta, TextDelta)) == (
            "Hello there, friend."
        )
        assert isinstance(done, CompletionDone)
        assert done.message.content == "Hello there, friend."
        assert done.message.model == "openai/gpt-4o"
        assert done.tool_calls == []

    async def test_should_assemble_tool_calls_split_across_chunks(
        self, provider: LiteLLMProvider, model_config: ModelConfig, user_messages: list[Message]
    ) -> None:
        chunks = [
            _stream_chunk(Delta(content="Let me check.")),
            _stream_chunk(_tool_call_fragment('{"expre', first=True)),
            _stream_chunk(_tool_call_fragment('ssion": "2+2"}', first=False)),
            _stream_chunk(Delta(), finish_reason="tool_calls"),
        ]
        schema = ToolSchema(name="calculator", description="Math.", parameters={})
        target = "miniagent.llm.litellm_provider.litellm.acompletion"
        with patch(target, AsyncMock(return_value=_replay(chunks))) as acompletion:
            events = await _collect(
                provider.stream_completion(user_messages, model_config, tools=[schema])
            )

        assert events[0] == TextDelta("Let me check.")
        done = events[-1]
        assert isinstance(done, CompletionDone)
        assert done.tool_calls == [
            ToolCall(id="call_1", tool_name="calculator", arguments={"expression": "2+2"})
        ]
        assert done.message.tool_calls == done.tool_calls
        request = acompletion.call_args.kwargs
        assert request["stream"] is True
        assert request["tool_choice"] == "auto"
        assert request["tools"][0]["function"]["name"] == "calculator"

    async def test_should_raise_provider_error_when_the_stream_fails(
        self, provider: LiteLLMProvider, model_config: ModelConfig, user_messages: list[Message]
    ) -> None:
        target = "miniagent.llm.litellm_provider.litellm.acompletion"
        with (
            patch(target, AsyncMock(side_effect=RuntimeError("connection reset"))),
            pytest.raises(LLMProviderError, match="connection reset"),
        ):
            await _collect(provider.stream_completion(user_messages, model_config))

    async def test_should_raise_provider_error_for_an_empty_stream(
        self, provider: LiteLLMProvider, model_config: ModelConfig, user_messages: list[Message]
    ) -> None:
        target = "miniagent.llm.litellm_provider.litellm.acompletion"
        with (
            patch(target, AsyncMock(return_value=_replay([]))),
            pytest.raises(LLMProviderError, match="empty stream"),
        ):
            await _collect(provider.stream_completion(user_messages, model_config))


# ---------------------------------------------------------------------------
# count_tokens() tests
# ---------------------------------------------------------------------------


def _tool_call_turn() -> list[Message]:
    call = ToolCall(id="c1", tool_name="calculator", arguments={"expression": "2+2"})
    return [
        Message(role=MessageRole.USER, content="What is 2 + 2?"),
        Message(role=MessageRole.AI, content="[tool call]", tool_calls=[call]),
        Message(role=MessageRole.TOOL, content="4", tool_call_id="c1", tool_name="calculator"),
    ]


class TestCountTokens:
    """LiteLLMProvider.count_tokens() uses LiteLLM's tokenizer for the model."""

    def test_should_pass_model_and_converted_messages_to_litellm(
        self, provider: LiteLLMProvider, model_config: ModelConfig
    ) -> None:
        messages = _tool_call_turn()
        target = "miniagent.llm.litellm_provider.litellm.token_counter"
        with patch(target, return_value=123) as token_counter:
            assert provider.count_tokens(messages, model_config) == 123
        token_counter.assert_called_once_with(
            model="openai/gpt-4o", messages=to_litellm_messages(messages)
        )

    def test_should_count_tool_calls_as_well_as_text(
        self, provider: LiteLLMProvider, model_config: ModelConfig
    ) -> None:
        messages = _tool_call_turn()
        without_call = [messages[0], Message(role=MessageRole.AI, content="[tool call]")]
        assert provider.count_tokens(messages[:2], model_config) > provider.count_tokens(
            without_call, model_config
        )

    def test_should_return_zero_for_no_messages(
        self, provider: LiteLLMProvider, model_config: ModelConfig
    ) -> None:
        assert provider.count_tokens([], model_config) == 0


# ---------------------------------------------------------------------------
# Provider registry tests
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    """get_default_provider, reset_default_provider, set_default_provider."""

    def setup_method(self) -> None:
        reset_default_provider()

    def teardown_method(self) -> None:
        reset_default_provider()

    def test_returns_litellm_provider(self) -> None:
        p = get_default_provider()
        assert isinstance(p, LiteLLMProvider)

    def test_returns_same_singleton(self) -> None:
        p1 = get_default_provider()
        p2 = get_default_provider()
        assert p1 is p2

    def test_reset_creates_new_instance(self) -> None:
        p1 = get_default_provider()
        reset_default_provider()
        p2 = get_default_provider()
        assert p1 is not p2

    def test_set_custom_provider(self) -> None:
        mock_provider = MagicMock(spec=LiteLLMProvider)
        set_default_provider(mock_provider)
        assert get_default_provider() is mock_provider

    def test_set_provider_raises_on_wrong_type(self) -> None:
        with pytest.raises(TypeError):
            set_default_provider("not_a_provider")  # type: ignore[arg-type]
