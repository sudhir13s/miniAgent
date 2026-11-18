"""Tests for the ReAct loop shared by Agent.run() and Agent.stream()."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from miniagent.agent.events import (
    AgentEvent,
    RunCompleted,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from miniagent.agent.react_loop import LoopSettings, run_react_loop
from miniagent.core.exceptions import AgentExecutionError
from miniagent.core.message import Message, MessageRole, ToolCall
from miniagent.core.state import State, WorkflowStatus
from miniagent.llm.streaming import StreamEvent
from miniagent.profile.model_config import ModelConfig
from miniagent.tool.base import BaseTool
from miniagent.tool.function_tool import tool
from tests.unit.llm.fakes import ScriptedProvider, text_turn, tool_turn

_MODEL = ModelConfig(id="openai/gpt-4o")


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def _settings(
    provider: ScriptedProvider,
    *,
    streaming: bool,
    tools: dict[str, BaseTool] | None = None,
    max_iterations: int = 5,
) -> LoopSettings:
    return LoopSettings(
        agent_name="loop-test",
        max_iterations=max_iterations,
        model_config=_MODEL,
        provider=provider,
        tools=tools or {},
        fit_context=list,
        streaming=streaming,
    )


async def _run(settings: LoopSettings) -> list[AgentEvent]:
    start = [Message(role=MessageRole.USER, content="What is 2 + 3?")]
    return [event async for event in run_react_loop(settings, State(messages=start), start)]


_ADD_CALL = ToolCall(id="c1", tool_name="add", arguments={"a": 2, "b": 3})


class TestStreamingLoop:
    async def test_should_stream_text_then_complete(self) -> None:
        events = await _run(_settings(ScriptedProvider(text_turn("Hel", "lo")), streaming=True))
        assert events[:2] == [TextDelta("Hel"), TextDelta("lo")]
        assert isinstance(events[-1], RunCompleted)
        assert events[-1].state.messages[-1].content == "Hello"

    async def test_should_report_tool_calls_and_results_in_order(self) -> None:
        provider = ScriptedProvider(tool_turn(_ADD_CALL), text_turn("It is 5."))
        events = await _run(_settings(provider, streaming=True, tools={"add": add}))

        kinds = [type(event).__name__ for event in events]
        assert kinds == ["ToolCallStarted", "ToolCallFinished", "TextDelta", "RunCompleted"]
        started, finished = events[0], events[1]
        assert isinstance(started, ToolCallStarted) and started.tool_call == _ADD_CALL
        assert isinstance(finished, ToolCallFinished) and finished.result.content == "5"

    async def test_should_send_tool_results_back_to_the_model(self) -> None:
        provider = ScriptedProvider(tool_turn(_ADD_CALL), text_turn("It is 5."))
        await _run(_settings(provider, streaming=True, tools={"add": add}))
        second_call = provider.received[1]
        assert [message.role for message in second_call][-2:] == [MessageRole.AI, MessageRole.TOOL]


class TestNonStreamingLoop:
    async def test_should_emit_no_text_deltas(self) -> None:
        events = await _run(_settings(ScriptedProvider(text_turn("Hel", "lo")), streaming=False))
        assert [type(event) for event in events] == [RunCompleted]

    async def test_should_produce_the_same_final_state_as_streaming(self) -> None:
        def script() -> ScriptedProvider:
            return ScriptedProvider(tool_turn(_ADD_CALL), text_turn("It is 5."))

        streamed = await _run(_settings(script(), streaming=True, tools={"add": add}))
        plain = await _run(_settings(script(), streaming=False, tools={"add": add}))
        assert isinstance(streamed[-1], RunCompleted) and isinstance(plain[-1], RunCompleted)
        contents = [
            [m.content for m in event.state.messages] for event in (streamed[-1], plain[-1])
        ]
        assert contents[0] == contents[1]


async def _final_status(state: State, provider: ScriptedProvider) -> WorkflowStatus:
    settings = _settings(provider, streaming=False, tools={"add": add}, max_iterations=1)
    events = [event async for event in run_react_loop(settings, state, list(state.messages))]
    completed = events[-1]
    assert isinstance(completed, RunCompleted)
    return completed.state.status


class TestRunStatus:
    """A standalone run marks itself completed; a workflow keeps ownership of status."""

    async def test_should_mark_a_standalone_run_completed(self) -> None:
        status = await _final_status(State(), ScriptedProvider(text_turn("done")))
        assert status == WorkflowStatus.COMPLETED

    async def test_should_mark_completed_even_when_iterations_run_out(self) -> None:
        status = await _final_status(State(), ScriptedProvider(tool_turn(_ADD_CALL)))
        assert status == WorkflowStatus.COMPLETED

    async def test_should_leave_a_workflow_managed_status_alone(self) -> None:
        running = State(status=WorkflowStatus.RUNNING)
        status = await _final_status(running, ScriptedProvider(text_turn("done")))
        assert status == WorkflowStatus.RUNNING


class TestLoopLimits:
    async def test_should_complete_with_current_state_at_max_iterations(self) -> None:
        provider = ScriptedProvider(tool_turn(_ADD_CALL), tool_turn(_ADD_CALL))
        events = await _run(
            _settings(provider, streaming=True, tools={"add": add}, max_iterations=2)
        )
        assert isinstance(events[-1], RunCompleted)
        assert events[-1].state.step_count == 4

    async def test_should_fail_when_the_provider_ends_a_turn_without_a_reply(self) -> None:
        class _SilentProvider(ScriptedProvider):
            async def stream_completion(
                self, *args: object, **kwargs: object
            ) -> AsyncIterator[StreamEvent]:
                yield TextDelta("half a reply")

        with pytest.raises(AgentExecutionError, match="without a reply"):
            await _run(_settings(_SilentProvider(), streaming=True))
