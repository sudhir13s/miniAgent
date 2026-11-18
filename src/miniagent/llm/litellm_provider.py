"""LiteLLM-backed implementation of BaseLLMProvider.

Routes ALL LLM calls through ``litellm.acompletion()``, giving a unified
interface for every supported provider — OpenAI, Anthropic, Google, Groq,
Bedrock, Azure, Vertex AI, and many more.

Model strings use the LiteLLM ``<provider>/<model>`` format::

    "openai/gpt-4o"
    "anthropic/claude-3-5-sonnet-20241022"
    "gemini/gemini-2.0-flash"
    "groq/llama-3.1-70b-versatile"
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"

API keys are read from environment variables following each provider's
convention (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``,
etc.).  LiteLLM handles all provider-specific authentication automatically.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, cast

import litellm
from litellm.types.utils import ModelResponse

from miniagent.core.exceptions import LLMProviderError
from miniagent.core.message import Message, ToolCall
from miniagent.llm.base import BaseLLMProvider
from miniagent.llm.litellm_format import (
    to_ai_turn,
    to_litellm_messages,
    to_litellm_parameters,
    to_litellm_tools,
)
from miniagent.llm.litellm_spec import TOOL_CHOICE_AUTO, RequestParam
from miniagent.llm.streaming import CompletionDone, StreamEvent, TextDelta

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponseStream

    from miniagent.profile.model_config import ModelConfig
    from miniagent.tool.base import ToolSchema

logger = logging.getLogger(__name__)


class LiteLLMProvider(BaseLLMProvider):
    """Universal LLM provider backed by LiteLLM.

    All completion requests are forwarded to ``litellm.acompletion()``.
    The provider is stateless — it holds no connections and can be shared
    safely across concurrent coroutines.

    Usage::

        provider = LiteLLMProvider()
        reply = await provider.complete(messages, model_config)
    """

    def count_tokens(self, messages: Sequence[Message], model_config: ModelConfig) -> int:
        """Count tokens with LiteLLM's tokenizer for the model.

        LiteLLM picks the model's own tokenizer where it has one and falls
        back to ``tiktoken`` otherwise; tool calls are counted too.

        Args:
            messages: The messages to count.
            model_config: The model whose tokenizer applies.

        Returns:
            The token count, ``0`` for no messages.
        """
        if not messages:
            return 0
        count: int = litellm.token_counter(
            model=model_config.id, messages=to_litellm_messages(messages)
        )
        return count

    async def complete(self, messages: list[Message], model_config: ModelConfig) -> Message:
        """Request a text completion from LiteLLM, without tools or streaming.

        Args:
            messages: Conversation history.
            model_config: Per-agent model settings.

        Returns:
            An AI ``Message`` with the LLM response text.

        Raises:
            LLMProviderError: If the LiteLLM call fails or the reply is empty.
        """
        logger.debug("LiteLLM complete: model=%s", model_config.id)
        response = await self._acompletion(self._request(messages, model_config, tools=None))
        message, _ = to_ai_turn(response.choices[0].message, model_config.id)
        return message

    async def complete_with_tools(
        self,
        messages: list[Message],
        model_config: ModelConfig,
        tools: list[ToolSchema],
    ) -> tuple[Message, list[ToolCall]]:
        """Request a completion that may include tool calls.

        Args:
            messages: Conversation history.
            model_config: Per-agent model settings.
            tools: Tool schemas available to the LLM.

        Returns:
            A tuple of (AI Message, list of ToolCall objects).

        Raises:
            LLMProviderError: If the LiteLLM call fails, the reply is empty,
                or a tool call does not name its tool.
        """
        logger.debug("LiteLLM complete_with_tools: model=%s tools=%d", model_config.id, len(tools))
        response = await self._acompletion(self._request(messages, model_config, tools))
        return to_ai_turn(response.choices[0].message, model_config.id)

    async def stream_completion(
        self,
        messages: list[Message],
        model_config: ModelConfig,
        tools: list[ToolSchema] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one turn: text deltas as they arrive, then the assembled turn.

        Tool-call fragments spread over many chunks are reassembled by
        ``litellm.stream_chunk_builder`` into the same message and tool calls
        ``complete_with_tools`` would return.

        Args:
            messages: Conversation history.
            model_config: Per-agent model settings.
            tools: Tool schemas the model may call; ``None`` or empty for none.

        Yields:
            ``TextDelta`` events, then exactly one ``CompletionDone``.

        Raises:
            LLMProviderError: If the stream fails or ends without any chunk,
                the reply is empty, or a tool call does not name its tool.
        """
        request = self._request(messages, model_config, tools)
        request[RequestParam.STREAM] = True
        logger.debug("LiteLLM stream_completion: model=%s", model_config.id)
        chunks: list[ModelResponseStream] = []
        async for chunk in self._stream_chunks(request):
            chunks.append(chunk)
            text = _delta_text(chunk)
            if text:
                yield TextDelta(text)
        response = litellm.stream_chunk_builder(chunks, messages=request[RequestParam.MESSAGES])
        # The builder returns None when there were no chunks to assemble.
        if not chunks or not isinstance(response, ModelResponse):
            raise LLMProviderError(f"LiteLLM returned an empty stream for '{model_config.id}'.")
        yield CompletionDone(*to_ai_turn(response.choices[0].message, model_config.id))

    def _request(
        self,
        messages: Sequence[Message],
        model_config: ModelConfig,
        tools: Sequence[ToolSchema] | None,
    ) -> dict[str, Any]:
        """Build the ``litellm.acompletion`` keyword arguments for one call."""
        request = to_litellm_parameters(model_config)
        request[RequestParam.MESSAGES] = to_litellm_messages(messages)
        if tools:
            request[RequestParam.TOOLS] = to_litellm_tools(tools)
            request[RequestParam.TOOL_CHOICE] = TOOL_CHOICE_AUTO
        return request

    async def _acompletion(self, request: dict[str, Any]) -> ModelResponse:
        """Call LiteLLM once, turning any failure into ``LLMProviderError``."""
        try:
            # LiteLLM declares a response-or-stream union; without stream=True
            # it is always the complete response.
            return cast("ModelResponse", await litellm.acompletion(**request))
        except Exception as exc:
            raise LLMProviderError(
                f"LiteLLM completion failed for model '{request[RequestParam.MODEL]}': {exc}",
                cause=exc,
            ) from exc

    async def _stream_chunks(self, request: dict[str, Any]) -> AsyncIterator[ModelResponseStream]:
        """Yield LiteLLM stream chunks, turning any failure into ``LLMProviderError``."""
        try:
            # LiteLLM declares a response-or-stream union; with stream=True it is
            # always the async chunk stream.
            stream = cast(
                "AsyncIterator[ModelResponseStream]", await litellm.acompletion(**request)
            )
            async for chunk in stream:
                yield chunk
        except Exception as exc:
            raise LLMProviderError(
                f"LiteLLM stream failed for model '{request[RequestParam.MODEL]}': {exc}",
                cause=exc,
            ) from exc


def _delta_text(chunk: ModelResponseStream) -> str:
    """Return the new reply text in a stream chunk, or ``""``."""
    if not chunk.choices:
        return ""
    delta = chunk.choices[0].delta
    return (delta.content or "") if delta else ""
