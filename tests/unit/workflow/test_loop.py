"""Tests for LoopWorkflow."""

from __future__ import annotations

import pytest

from miniagent.agent.base import BaseAgent
from miniagent.core.exceptions import WorkflowExecutionError
from miniagent.core.message import MessageRole
from miniagent.core.state import State, WorkflowStatus
from miniagent.workflow.loop import LoopWorkflow
from tests.unit.workflow.fakes import (
    AppendAgent,
    CounterAgent,
    make_profile,
)


class TestLoopWorkflow:
    """Tests for LoopWorkflow."""

    async def test_runs_max_iterations_without_exit_condition(self) -> None:
        agent = CounterAgent(make_profile("counter"))
        wf = LoopWorkflow("loop", max_iterations=3)
        wf.register_agent(agent)
        final = await wf.run(State(max_steps=200))
        assert final.variables["count"] == 3

    async def test_exit_condition_stops_loop_early(self) -> None:
        agent = CounterAgent(make_profile("counter"))

        def stop_at_2(state: State) -> bool:
            return state.variables.get("count", 0) >= 2

        wf = LoopWorkflow("loop", exit_condition=stop_at_2, max_iterations=10)
        wf.register_agent(agent)
        final = await wf.run(State(max_steps=200))
        assert final.variables["count"] == 2

    async def test_exit_condition_on_first_iteration(self) -> None:
        agent = CounterAgent(make_profile("counter"))

        wf = LoopWorkflow("loop", exit_condition=lambda s: True, max_iterations=5)
        wf.register_agent(agent)
        final = await wf.run(State(max_steps=200))
        # Should have run exactly once (exit immediately after first iter)
        assert final.variables["count"] == 1

    async def test_fresh_state_created_when_none(self) -> None:
        agent = CounterAgent(make_profile("counter"))
        wf = LoopWorkflow("loop", max_iterations=1)
        wf.register_agent(agent)
        final = await wf.run()
        assert final.status == WorkflowStatus.COMPLETED

    async def test_status_completed_after_max_iterations(self) -> None:
        agent = CounterAgent(make_profile("counter"))
        wf = LoopWorkflow("loop", max_iterations=2)
        wf.register_agent(agent)
        final = await wf.run(State(max_steps=200))
        assert final.status == WorkflowStatus.COMPLETED

    async def test_raises_when_no_agents(self) -> None:
        wf = LoopWorkflow("empty", max_iterations=3)
        with pytest.raises(WorkflowExecutionError, match="no agents"):
            await wf.run()

    async def test_raises_when_agent_not_registered(self) -> None:
        wf = LoopWorkflow("bad", agent_order=["ghost"], max_iterations=1)
        with pytest.raises(WorkflowExecutionError, match="not registered"):
            await wf.run()

    async def test_raises_on_invalid_max_iterations(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            LoopWorkflow("bad", max_iterations=0)

    async def test_agent_error_wraps_in_workflow_error(self) -> None:
        class _BrokenAgent(BaseAgent):
            async def run(self, state: State) -> State:
                raise RuntimeError("broken")

        agent = _BrokenAgent(make_profile("broken"))
        wf = LoopWorkflow("loop", max_iterations=3)
        wf.register_agent(agent)
        with pytest.raises(WorkflowExecutionError):
            await wf.run(State(max_steps=200))

    async def test_multiple_agents_per_iteration(self) -> None:
        """Each iteration runs agent-a then agent-b sequentially."""
        a = AppendAgent(make_profile("a"), "A")
        b = AppendAgent(make_profile("b"), "B")
        wf = LoopWorkflow("loop", agent_order=["a", "b"], max_iterations=2)
        wf.register_agent(a)
        wf.register_agent(b)
        final = await wf.run(State(max_steps=200))
        ai_msgs = [m.content for m in final.messages if m.role == MessageRole.AI]
        # 2 iterations x 2 agents = 4 AI messages
        assert len(ai_msgs) == 4
