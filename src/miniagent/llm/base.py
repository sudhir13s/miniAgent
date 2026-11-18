"""Abstract base class defining the contract for LLM providers.

All LLM calls in miniAgent route through a BaseLLMProvider implementation.
The default implementation is LiteLLMProvider which supports every major
provider (OpenAI, Anthropic, Google, Groq, Bedrock, Azure, etc.) via a
unified model string like ``"openai/gpt-4o"`` or ``"gemini/gemini-2.0-flash"``.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING

from miniagent.llm.streaming import CompletionDone, StreamEvent, TextDelta

if TYPE_CHECKING:
    from miniagent.core.message import Message, ToolCall
    from miniagent.profile.model_config import ModelConfig
    from miniagent.tool.base import ToolSchema

# Rough averages for English text and the per-message framing most chat
# formats add; used only when a provider has no tokenizer of its own.
_CHARACTERS_PER_TOKEN = 4
_TOKENS_PER_MESSAGE = 4


class BaseLLMProvider(ABC):
    """Abstract gateway for all LLM completions.

    Concrete implementations (e.g. ``LiteLLMProvider``) translate the
    framework's internal message format into provider-specific API calls
    and return standardised ``Message`` objects.

    A provider must implement ``complete`` and ``complete_with_tools``.
    ``stream_completion`` and ``count_tokens`` have working defaults that a
    provider overrides to stream for real and to count tokens exactly.
    """

    @abstractmethod
    async def complete(self, messages: list[Message], model_config: ModelConfig) -> Message:
        """Request a text completion from the LLM, without tools.

        Args:
            messages: Conversation history in miniAgent ``Message`` format.
            model_config: Per-agent model settings (model string, temperature,
                max_tokens, etc.).

        Returns:
            An AI ``Message`` containing the LLM's response text.

        Raises:
            LLMProviderError: If the provider call fails (rate limit, auth,
                timeout, unsupported model).
        """

    @abstractmethod
    async def complete_with_tools(
        self,
        messages: list[Message],
        model_config: ModelConfig,
        tools: list[ToolSchema],
    ) -> tuple[Message, list[ToolCall]]:
        """Request a completion that may include tool calls.

        Sends messages plus tool schemas to the LLM and parses any tool-call
        requests from the response.

        Args:
            messages: Conversation history.
            model_config: Per-agent model settings.
            tools: Tool schemas available to the LLM.

        Returns:
            A tuple of:
            - The AI ``Message``; a turn with only tool calls holds a
              placeholder as its content.
            - A list of ``ToolCall`` objects parsed from the response.
              Empty list if the model produced only text.

        Raises:
            LLMProviderError: If the provider call fails.
        """

    async def stream_completion(
        self,
        messages: list[Message],
        model_config: ModelConfig,
        tools: list[ToolSchema] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one model turn: text as it is written, then the finished turn.

        Yields ``TextDelta`` events followed by one ``CompletionDone``. This
        default does not stream: it makes a normal completion and yields a
        text-only reply as a single delta. Providers that can stream should
        override it.

        Args:
            messages: Conversation history.
            model_config: Per-agent model settings.
            tools: Tool schemas the model may call; ``None`` or empty for none.

        Yields:
            ``TextDelta`` events, then exactly one ``CompletionDone``.

        Raises:
            LLMProviderError: If the provider call fails.
        """
        if tools:
            message, tool_calls = await self.complete_with_tools(messages, model_config, tools)
        else:
            message, tool_calls = await self.complete(messages, model_config), []
        if not tool_calls and message.content:
            yield TextDelta(message.content)
        yield CompletionDone(message, tool_calls)

    def count_tokens(self, messages: Sequence[Message], model_config: ModelConfig) -> int:
        """Count the tokens ``messages`` cost for ``model_config``'s model.

        Context window policies use this to fit history into a budget. The
        default is an estimate of about four characters per token, including
        any tool calls an AI message requested; providers with a real
        tokenizer should override it.

        Args:
            messages: The messages to count.
            model_config: The model whose tokenizer applies.

        Returns:
            The token count, ``0`` for no messages.
        """
        return sum(_estimate_message_tokens(message) for message in messages)


def _estimate_message_tokens(message: Message) -> int:
    """Estimate one message's tokens from its text and requested tool calls."""
    text = message.content
    if message.tool_calls:
        text += json.dumps([call.model_dump() for call in message.tool_calls])
    return _TOKENS_PER_MESSAGE + math.ceil(len(text) / _CHARACTERS_PER_TOKEN)
