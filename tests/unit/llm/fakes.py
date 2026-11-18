"""A scripted LLM provider for agent, loop and API tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from miniagent.core.message import Message, MessageRole, ToolCall
from miniagent.llm.base import BaseLLMProvider
from miniagent.llm.streaming import CompletionDone, StreamEvent, TextDelta
from miniagent.profile.model_config import ModelConfig
from miniagent.tool.base import ToolSchema


@dataclass(frozen=True)
class ScriptedTurn:
    """One model turn: reply text (streamed in these pieces) and tool calls."""

    text_pieces: Sequence[str] = ()
    tool_calls: Sequence[ToolCall] = field(default_factory=tuple)

    @property
    def message(self) -> Message:
        content = "".join(self.text_pieces) or "[tool call]"
        return Message(role=MessageRole.AI, content=content, tool_calls=list(self.tool_calls))


def text_turn(*pieces: str) -> ScriptedTurn:
    return ScriptedTurn(text_pieces=pieces)


def tool_turn(*tool_calls: ToolCall) -> ScriptedTurn:
    return ScriptedTurn(tool_calls=tool_calls)


class ScriptedProvider(BaseLLMProvider):
    """Plays back turns in order and records the messages each call received."""

    def __init__(self, *turns: ScriptedTurn) -> None:
        self._turns = list(turns)
        self.received: list[list[Message]] = []

    def _next(self, messages: list[Message]) -> ScriptedTurn:
        self.received.append(list(messages))
        return self._turns.pop(0)

    async def complete(self, messages: list[Message], model_config: ModelConfig) -> Message:
        return self._next(messages).message

    async def complete_with_tools(
        self,
        messages: list[Message],
        model_config: ModelConfig,
        tools: list[ToolSchema],
    ) -> tuple[Message, list[ToolCall]]:
        turn = self._next(messages)
        return turn.message, list(turn.tool_calls)

    async def stream_completion(
        self,
        messages: list[Message],
        model_config: ModelConfig,
        tools: list[ToolSchema] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        turn = self._next(messages)
        for piece in turn.text_pieces:
            yield TextDelta(piece)
        yield CompletionDone(turn.message, list(turn.tool_calls))
