"""Events a provider yields while streaming one model turn.

A streamed turn is any number of ``TextDelta`` events, as the model writes,
followed by exactly one ``CompletionDone`` holding the finished message and
any tool calls the model requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from miniagent.core.message import Message, ToolCall


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A piece of reply text, in the order the model produced it.

    Attributes:
        text: The new text; join every delta of a turn to get its full text.
    """

    text: str


@dataclass(frozen=True, slots=True)
class CompletionDone:
    """The end of a streamed turn.

    Attributes:
        message: The complete AI message, as ``complete_with_tools`` returns it.
        tool_calls: Tools the model requested; empty for a text-only reply.
    """

    message: Message
    tool_calls: list[ToolCall] = field(default_factory=list)


StreamEvent = TextDelta | CompletionDone
