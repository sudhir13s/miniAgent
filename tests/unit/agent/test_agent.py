"""Tests for Agent — the universal ReAct agent.

All LLM calls are mocked via an injected mock BaseLLMProvider so no
real API calls are made.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from miniagent.agent.agent import Agent
from miniagent.agent.base import BaseAgent
from miniagent.agent.events import AgentEvent, RunCompleted, TextDelta
from miniagent.core.exceptions import AgentExecutionError, ToolInvocationError
from miniagent.core.message import Message, MessageRole, ToolCall
from miniagent.core.state import State
from miniagent.llm.streaming import StreamEvent
from miniagent.memory.context_window import TokenBudgetPolicy
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.model_config import ModelConfig
from miniagent.tool.base import BaseTool, ToolSchema
from tests.unit.llm.fakes import ScriptedProvider, text_turn
from tests.unit.tool.fakes import FakeToolset

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _profile(name: str = "test-agent", max_iterations: int = 5) -> AgentProfile:
    return AgentProfile(
        name=name,
        model=ModelConfig(id="openai/gpt-4o"),
        system_prompt="You are a test assistant.",
        max_iterations=max_iterations,
    )


def _mock_provider(
    content: str = "Hello!",
    tool_calls: list[ToolCall] | None = None,
) -> MagicMock:
    """Build a mock BaseLLMProvider."""
    provider = MagicMock()
    ai_msg = Message(role=MessageRole.AI, content=content)

    if tool_calls is None:
        # complete() and complete_with_tools() both return just text
        provider.complete = AsyncMock(return_value=ai_msg)
        provider.complete_with_tools = AsyncMock(return_value=(ai_msg, []))
    else:
        # First call returns tool calls, second call returns text
        text_msg = Message(role=MessageRole.AI, content="Done!")
        provider.complete = AsyncMock(return_value=text_msg)
        provider.complete_with_tools = AsyncMock(side_effect=[(ai_msg, tool_calls), (text_msg, [])])
    return provider


class _EchoTool(BaseTool):
    """Simple tool that echoes its input."""

    def __init__(self) -> None:
        super().__init__(
            ToolSchema(
                name="echo",
                description="Echo input back",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            )
        )

    async def invoke(self, **kwargs: Any) -> str:
        return f"echo: {kwargs.get('text', '')}"


class _FailTool(BaseTool):
    """Tool that always raises ToolInvocationError."""

    def __init__(self) -> None:
        super().__init__(ToolSchema(name="fail", description="Always fails", parameters={}))

    async def invoke(self, **kwargs: Any) -> str:
        raise ToolInvocationError("Tool deliberately failed")


class _CrashTool(BaseTool):
    """Tool with a bug: raises an unexpected, non-framework exception."""

    def __init__(self) -> None:
        super().__init__(ToolSchema(name="crash", description="Has a bug", parameters={}))

    async def invoke(self, **kwargs: Any) -> str:
        raise KeyError("missing_key")


class _SlowTool(BaseTool):
    """Tool that sleeps longer than its own timeout."""

    timeout_seconds: float | None = 0.01

    def __init__(self) -> None:
        super().__init__(ToolSchema(name="slow", description="Too slow", parameters={}))

    async def invoke(self, **kwargs: Any) -> str:
        await asyncio.sleep(1)
        return "never returned"


class _ConcurrencyProbeTool(BaseTool):
    """Records how many invocations overlap, to prove tool calls run concurrently."""

    def __init__(self, name: str, delay_seconds: float, tracker: dict[str, int]) -> None:
        super().__init__(ToolSchema(name=name, description="Probe", parameters={}))
        self._delay_seconds = delay_seconds
        self._tracker = tracker

    async def invoke(self, **kwargs: Any) -> str:
        self._tracker["running"] += 1
        self._tracker["peak"] = max(self._tracker["peak"], self._tracker["running"])
        await asyncio.sleep(self._delay_seconds)
        self._tracker["running"] -= 1
        return f"{self.name} done"


def _tool_messages(state: State) -> list[Message]:
    return [message for message in state.messages if message.role == MessageRole.TOOL]


@pytest.fixture()
def initial_state() -> State:
    return State(
        messages=[Message(role=MessageRole.USER, content="Say hello")],
        max_steps=50,
    )


# ---------------------------------------------------------------------------
# run() — no tools
# ---------------------------------------------------------------------------


class TestExecuteNoTools:
    """run() when the agent has no tools."""

    async def test_appends_ai_message(self, initial_state: State) -> None:
        provider = _mock_provider("Hi there!")
        agent = Agent(_profile(), provider)
        new_state = await agent.run(initial_state)
        ai_msgs = [m for m in new_state.messages if m.role == MessageRole.AI]
        assert len(ai_msgs) == 1
        assert ai_msgs[0].content == "Hi there!"

    async def test_increments_step_count(self, initial_state: State) -> None:
        provider = _mock_provider("Hello")
        agent = Agent(_profile(), provider)
        new_state = await agent.run(initial_state)
        assert new_state.step_count == 1

    async def test_system_prompt_added_automatically(self, initial_state: State) -> None:
        provider = _mock_provider("Hello")
        agent = Agent(_profile(), provider)
        await agent.run(initial_state)
        call_messages = provider.complete.call_args.args[0]
        assert call_messages[0].role == MessageRole.SYSTEM
        assert "test assistant" in call_messages[0].content

    async def test_system_prompt_not_duplicated_when_already_present(self) -> None:
        provider = _mock_provider("Hello")
        agent = Agent(_profile(), provider)
        state = State(
            messages=[
                Message(role=MessageRole.SYSTEM, content="Custom system"),
                Message(role=MessageRole.USER, content="Hi"),
            ],
            max_steps=50,
        )
        await agent.run(state)
        call_messages = provider.complete.call_args.args[0]
        system_msgs = [m for m in call_messages if m.role == MessageRole.SYSTEM]
        assert len(system_msgs) == 1

    async def test_raises_agent_execution_error_on_provider_failure(
        self, initial_state: State
    ) -> None:
        provider = MagicMock()
        provider.complete = AsyncMock(side_effect=RuntimeError("API down"))
        agent = Agent(_profile(), provider)
        with pytest.raises(AgentExecutionError):
            await agent.run(initial_state)


# ---------------------------------------------------------------------------
# run() — with tools
# ---------------------------------------------------------------------------


class TestExecuteWithTools:
    """run() when the agent has tools registered."""

    async def test_invokes_tool_and_appends_result(self, initial_state: State) -> None:
        tool_call = ToolCall(id="tc1", tool_name="echo", arguments={"text": "hi"})
        provider = _mock_provider("Initial", tool_calls=[tool_call])
        echo = _EchoTool()
        agent = Agent(_profile(), provider, tools={"echo": echo})
        new_state = await agent.run(initial_state)

        tool_msgs = [m for m in new_state.messages if m.role == MessageRole.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "echo: hi"

    async def test_step_count_includes_tool_invocations(self, initial_state: State) -> None:
        tool_call = ToolCall(id="tc1", tool_name="echo", arguments={"text": "x"})
        provider = _mock_provider("msg", tool_calls=[tool_call])
        agent = Agent(_profile(), provider, tools={"echo": _EchoTool()})
        new_state = await agent.run(initial_state)
        # 1 AI msg step + 1 tool msg step + 1 final AI msg step = 3
        assert new_state.step_count == 3

    async def test_unknown_tool_returns_error_message_not_raises(
        self, initial_state: State
    ) -> None:
        # Agent has "echo" registered but LLM requests "unknown_tool"
        tool_call = ToolCall(id="tc1", tool_name="unknown_tool", arguments={})
        provider = _mock_provider("msg", tool_calls=[tool_call])
        agent = Agent(_profile(), provider, tools={"echo": _EchoTool()})
        new_state = await agent.run(initial_state)
        tool_msgs = [m for m in new_state.messages if m.role == MessageRole.TOOL]
        assert any("not registered" in m.content for m in tool_msgs)

    async def test_tool_error_is_returned_to_model_and_run_continues(
        self, initial_state: State
    ) -> None:
        tool_call = ToolCall(id="tc1", tool_name="fail", arguments={})
        provider = _mock_provider("msg", tool_calls=[tool_call])
        agent = Agent(_profile(), provider, tools={"fail": _FailTool()})

        new_state = await agent.run(initial_state)

        [tool_message] = _tool_messages(new_state)
        assert tool_message.is_error is True
        assert tool_message.tool_call_id == "tc1"
        assert "Tool deliberately failed" in tool_message.content
        assert new_state.messages[-1].content == "Done!"

    async def test_unexpected_tool_exception_is_returned_to_model(
        self, initial_state: State
    ) -> None:
        tool_call = ToolCall(id="tc1", tool_name="crash", arguments={})
        provider = _mock_provider("msg", tool_calls=[tool_call])
        agent = Agent(_profile(), provider, tools={"crash": _CrashTool()})

        new_state = await agent.run(initial_state)

        [tool_message] = _tool_messages(new_state)
        assert tool_message.is_error is True
        assert "KeyError" in tool_message.content
        assert "missing_key" in tool_message.content

    async def test_invalid_arguments_are_returned_to_model(self, initial_state: State) -> None:
        """A hallucinated argument name must not crash the run; the model can retry."""
        tool_call = ToolCall(id="tc1", tool_name="slow", arguments={"bogus": 1})
        provider = _mock_provider("msg", tool_calls=[tool_call])

        class _NoArgsTool(BaseTool):
            def __init__(self) -> None:
                super().__init__(ToolSchema(name="slow", description="No args", parameters={}))

            async def invoke(self) -> str:  # type: ignore[override]
                return "ok"

        agent = Agent(_profile(), provider, tools={"slow": _NoArgsTool()})

        new_state = await agent.run(initial_state)

        [tool_message] = _tool_messages(new_state)
        assert tool_message.is_error is True
        assert "bogus" in tool_message.content

    async def test_tool_timeout_is_returned_to_model(self, initial_state: State) -> None:
        tool_call = ToolCall(id="tc1", tool_name="slow", arguments={})
        provider = _mock_provider("msg", tool_calls=[tool_call])
        agent = Agent(_profile(), provider, tools={"slow": _SlowTool()})

        new_state = await agent.run(initial_state)

        [tool_message] = _tool_messages(new_state)
        assert tool_message.is_error is True
        assert "timed out after 0.01 seconds" in tool_message.content

    async def test_tool_calls_in_one_turn_run_concurrently(self, initial_state: State) -> None:
        tracker = {"running": 0, "peak": 0}
        tools: dict[str, BaseTool] = {
            "first": _ConcurrencyProbeTool("first", 0.02, tracker),
            "second": _ConcurrencyProbeTool("second", 0.02, tracker),
        }
        tool_calls = [
            ToolCall(id="tc1", tool_name="first", arguments={}),
            ToolCall(id="tc2", tool_name="second", arguments={}),
        ]
        agent = Agent(_profile(), _mock_provider("msg", tool_calls=tool_calls), tools=tools)

        await agent.run(initial_state)

        assert tracker["peak"] == 2

    async def test_tool_results_keep_the_models_call_order(self, initial_state: State) -> None:
        """The slower first call still produces the first result message."""
        tracker = {"running": 0, "peak": 0}
        tools: dict[str, BaseTool] = {
            "slow_first": _ConcurrencyProbeTool("slow_first", 0.05, tracker),
            "fast_second": _ConcurrencyProbeTool("fast_second", 0.0, tracker),
        }
        tool_calls = [
            ToolCall(id="tc1", tool_name="slow_first", arguments={}),
            ToolCall(id="tc2", tool_name="fast_second", arguments={}),
        ]
        agent = Agent(_profile(), _mock_provider("msg", tool_calls=tool_calls), tools=tools)

        new_state = await agent.run(initial_state)

        assert [m.tool_call_id for m in _tool_messages(new_state)] == ["tc1", "tc2"]

    async def test_cancellation_is_not_swallowed(self, initial_state: State) -> None:
        """Cancelling the run must cancel it, not turn into a tool error message."""
        tool_call = ToolCall(id="tc1", tool_name="slow", arguments={})
        provider = _mock_provider("msg", tool_calls=[tool_call])

        class _ForeverTool(_SlowTool):
            timeout_seconds = None

        agent = Agent(_profile(), provider, tools={"slow": _ForeverTool()})
        run_task = asyncio.create_task(agent.run(initial_state))
        await asyncio.sleep(0.01)
        run_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await run_task

    async def test_max_iterations_stops_loop(self, initial_state: State) -> None:
        """When every LLM call returns tool calls, the loop stops at max_iterations."""
        tool_call = ToolCall(id="tc1", tool_name="echo", arguments={"text": "loop"})
        ai_msg = Message(role=MessageRole.AI, content="thinking")
        provider = MagicMock()
        provider.complete_with_tools = AsyncMock(return_value=(ai_msg, [tool_call]))
        agent = Agent(_profile(max_iterations=2), provider, tools={"echo": _EchoTool()})
        new_state = await agent.run(initial_state)
        # Should not raise — loop is capped at max_iterations
        assert new_state.step_count > 0


# ---------------------------------------------------------------------------
# run() — with toolsets
# ---------------------------------------------------------------------------


class TestRunWithToolsets:
    """run() resolves toolset tools at the start of every run."""

    async def test_toolset_tool_is_offered_and_invoked(self, initial_state: State) -> None:
        tool_call = ToolCall(id="tc1", tool_name="search", arguments={})
        provider = _mock_provider("msg", tool_calls=[tool_call])
        agent = Agent(_profile(), provider, toolsets=[FakeToolset("docs", ["search"])])

        new_state = await agent.run(initial_state)

        offered = provider.complete_with_tools.call_args_list[0].args[2]
        assert [schema.name for schema in offered] == ["search"]
        assert _tool_messages(new_state)[0].content == "search ran"

    async def test_tools_are_re_resolved_on_each_run(self, initial_state: State) -> None:
        toolset = FakeToolset("docs", ["search"])
        provider = _mock_provider("Hi")
        agent = Agent(_profile(), provider, toolsets=[toolset])

        await agent.run(initial_state)
        toolset.tool_names = ["search", "fetch"]
        await agent.run(initial_state)

        offered = provider.complete_with_tools.call_args_list[-1].args[2]
        assert [schema.name for schema in offered] == ["search", "fetch"]

    async def test_duplicate_tool_names_across_sources_fail_the_run(
        self, initial_state: State
    ) -> None:
        agent = Agent(
            _profile(),
            _mock_provider("Hi"),
            tools={"echo": _EchoTool()},
            toolsets=[FakeToolset("docs", ["echo"])],
        )

        with pytest.raises(AgentExecutionError, match=r"'echo'.*'docs'"):
            await agent.run(initial_state)


# ---------------------------------------------------------------------------
# run() — context window
# ---------------------------------------------------------------------------


def _ten_tokens_per_message(messages: list[Message], model_config: ModelConfig) -> int:
    return 10 * len(messages)


def _long_conversation() -> State:
    turns = ["q1", "a1", "q2", "a2", "q3"]
    roles = [MessageRole.USER, MessageRole.AI] * 2 + [MessageRole.USER]
    return State(
        messages=[
            Message(role=role, content=text) for role, text in zip(roles, turns, strict=True)
        ],
        max_steps=50,
    )


class TestRunWithContextPolicy:
    """run() sends the policy's window to the model but keeps the full history."""

    async def test_model_receives_only_the_window(self) -> None:
        provider = _mock_provider("a3")
        provider.count_tokens = MagicMock(side_effect=_ten_tokens_per_message)
        agent = Agent(_profile(), provider, context_policy=TokenBudgetPolicy(max_tokens=40))

        await agent.run(_long_conversation())

        sent = provider.complete.call_args.args[0]
        assert [message.content for message in sent] == [
            "You are a test assistant.",
            "q2",
            "a2",
            "q3",
        ]

    async def test_state_keeps_the_full_history(self) -> None:
        provider = _mock_provider("a3")
        provider.count_tokens = MagicMock(side_effect=_ten_tokens_per_message)
        agent = Agent(_profile(), provider, context_policy=TokenBudgetPolicy(max_tokens=40))

        new_state = await agent.run(_long_conversation())

        assert [message.content for message in new_state.messages] == [
            "q1",
            "a1",
            "q2",
            "a2",
            "q3",
            "a3",
        ]

    async def test_tokens_are_counted_for_the_agents_model(self) -> None:
        provider = _mock_provider("a3")
        provider.count_tokens = MagicMock(side_effect=_ten_tokens_per_message)
        agent = Agent(_profile(), provider, context_policy=TokenBudgetPolicy(max_tokens=40))

        await agent.run(_long_conversation())

        assert all(
            call.args[1] is agent.profile.model for call in provider.count_tokens.call_args_list
        )

    async def test_window_is_recomputed_after_each_tool_round(self) -> None:
        tool_call = ToolCall(id="tc1", tool_name="echo", arguments={"text": "x"})
        provider = _mock_provider("[tool call]", tool_calls=[tool_call])
        provider.count_tokens = MagicMock(side_effect=_ten_tokens_per_message)
        agent = Agent(
            _profile(),
            provider,
            tools={"echo": _EchoTool()},
            context_policy=TokenBudgetPolicy(max_tokens=50),
        )

        await agent.run(_long_conversation())

        first, second = (call.args[0] for call in provider.complete_with_tools.call_args_list)
        assert [message.content for message in first][-1] == "q3"
        assert [message.content for message in second][-2:] == ["[tool call]", "echo: x"]
        assert "q3" in [message.content for message in second]

    async def test_without_policy_everything_is_sent_and_nothing_is_counted(self) -> None:
        provider = _mock_provider("a3")
        provider.count_tokens = MagicMock()
        agent = Agent(_profile(), provider)

        await agent.run(_long_conversation())

        assert len(provider.complete.call_args.args[0]) == 6
        provider.count_tokens.assert_not_called()


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


async def _stream_events(agent: BaseAgent, state: State) -> list[AgentEvent]:
    return [event async for event in agent.stream(state)]


class TestStream:
    """stream() yields the run as events and ends with the final state."""

    async def test_streams_text_and_ends_with_the_final_state(self, initial_state: State) -> None:
        agent = Agent(_profile(), ScriptedProvider(text_turn("Hi ", "there!")))

        events = await _stream_events(agent, initial_state)

        assert events[:2] == [TextDelta("Hi "), TextDelta("there!")]
        completed = events[-1]
        assert isinstance(completed, RunCompleted)
        assert [m.content for m in completed.state.messages] == ["Say hello", "Hi there!"]

    async def test_applies_the_system_prompt_and_context_policy(self) -> None:
        provider = ScriptedProvider(text_turn("a3"))
        provider.count_tokens = _ten_tokens_per_message  # type: ignore[method-assign,assignment]
        agent = Agent(_profile(), provider, context_policy=TokenBudgetPolicy(max_tokens=40))

        await _stream_events(agent, _long_conversation())

        assert [m.content for m in provider.received[0]] == [
            "You are a test assistant.",
            "q2",
            "a2",
            "q3",
        ]

    async def test_wraps_provider_failures(self, initial_state: State) -> None:
        class _BrokenProvider(ScriptedProvider):
            async def stream_completion(
                self, *args: object, **kwargs: object
            ) -> AsyncIterator[StreamEvent]:
                raise RuntimeError("socket closed")
                yield  # pragma: no cover - makes this an async generator

        agent = Agent(_profile(), _BrokenProvider())

        with pytest.raises(AgentExecutionError, match="socket closed"):
            await _stream_events(agent, initial_state)

    async def test_custom_agents_stream_their_run_in_one_event(self, initial_state: State) -> None:
        class _EchoAgent(BaseAgent):
            async def run(self, state: State) -> State:
                return state.add_message(Message(role=MessageRole.AI, content="echo"))

        [event] = await _stream_events(_EchoAgent(_profile()), initial_state)

        assert isinstance(event, RunCompleted)
        assert event.state.messages[-1].content == "echo"
