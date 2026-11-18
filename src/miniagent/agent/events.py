"""Events ``agent.stream()`` yields while a run is in progress.

A run streams any mix of ``TextDelta``, ``ToolCallStarted`` and
``ToolCallFinished`` events and always ends with one ``RunCompleted``::

    async for event in agent.stream(state):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, RunCompleted):
            final_state = event.state
"""

from __future__ import annotations

from dataclasses import dataclass

from miniagent.core.message import Message, ToolCall
from miniagent.core.state import State
from miniagent.llm.streaming import TextDelta


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    """The model asked for a tool; it is about to run.

    Attributes:
        tool_call: The requested call: tool name and arguments.
    """

    tool_call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolCallFinished:
    """A tool finished; its result goes back to the model.

    Attributes:
        result: The TOOL message; its ``tool_call_id`` names the call it
            answers and ``is_error`` says whether the tool failed.
    """

    result: Message


@dataclass(frozen=True, slots=True)
class RunCompleted:
    """The run is over.

    Attributes:
        state: The final state, exactly what ``agent.run()`` returns.
    """

    state: State


AgentEvent = TextDelta | ToolCallStarted | ToolCallFinished | RunCompleted

__all__ = ["AgentEvent", "RunCompleted", "TextDelta", "ToolCallFinished", "ToolCallStarted"]
