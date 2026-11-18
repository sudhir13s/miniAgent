"""Tests for ParallelWorkflow."""

from __future__ import annotations

import pytest

from miniagent.agent.base import BaseAgent
from miniagent.core.exceptions import WorkflowExecutionError
from miniagent.core.message import Message, MessageRole
from miniagent.core.state import State, WorkflowStatus
from miniagent.profile.agent_profile import AgentProfile
from miniagent.workflow.parallel import ParallelWorkflow
from tests.unit.workflow.fakes import (
    AppendAgent,
    make_profile,
)


class TestParallelWorkflow:
    """Tests for ParallelWorkflow."""

    def _make_workflow(self, *agents: BaseAgent) -> ParallelWorkflow:
        wf = ParallelWorkflow("test-parallel")
        for agent in agents:
            wf.register_agent(agent)
        return wf

    async def test_all_agents_run(self) -> None:
        a = AppendAgent(make_profile("a"), "A")
        b = AppendAgent(make_profile("b"), "B")
        wf = self._make_workflow(a, b)
        final = await wf.run()
        ai_contents = [m.content for m in final.messages if m.role == MessageRole.AI]
        assert "response-A" in ai_contents
        assert "response-B" in ai_contents

    async def test_fresh_state_created_when_none(self) -> None:
        a = AppendAgent(make_profile("a"), "A")
        wf = self._make_workflow(a)
        final = await wf.run()
        assert len(final.messages) >= 1

    async def test_messages_from_all_branches_in_final_state(self) -> None:
        agents = [AppendAgent(make_profile(f"agent{i}"), f"label{i}") for i in range(3)]
        wf = self._make_workflow(*agents)
        final = await wf.run(State())
        assert len([m for m in final.messages if m.role == MessageRole.AI]) == 3

    async def test_step_count_is_sum_of_all_branches(self) -> None:
        a = AppendAgent(make_profile("a"), "A")
        b = AppendAgent(make_profile("b"), "B")
        wf = self._make_workflow(a, b)
        final = await wf.run(State())
        # Each branch does 1 step
        assert final.step_count == 2

    async def test_step_count_adds_branch_steps_to_the_starting_count(self) -> None:
        a = AppendAgent(make_profile("a"), "A")
        b = AppendAgent(make_profile("b"), "B")
        final = await self._make_workflow(a, b).run(State(step_count=5))
        assert final.step_count == 7

    async def test_variables_merged_from_all_branches(self) -> None:
        class _VarAgent(BaseAgent):
            def __init__(self, p: AgentProfile, key: str, val: str) -> None:
                super().__init__(p)
                self.key = key
                self.val = val

            async def run(self, state: State) -> State:
                msg = Message(role=MessageRole.AI, content=f"set {self.key}")
                return state.add_message(msg).increment_step().set_variable(self.key, self.val)

        a = _VarAgent(make_profile("a"), "x", "1")
        b = _VarAgent(make_profile("b"), "y", "2")
        wf = self._make_workflow(a, b)
        final = await wf.run(State())
        assert final.variables["x"] == "1"
        assert final.variables["y"] == "2"

    async def test_status_is_completed(self) -> None:
        a = AppendAgent(make_profile("a"), "A")
        wf = self._make_workflow(a)
        final = await wf.run(State())
        assert final.status == WorkflowStatus.COMPLETED

    async def test_raises_when_no_agents(self) -> None:
        wf = ParallelWorkflow("empty")
        with pytest.raises(WorkflowExecutionError, match="no agents"):
            await wf.run()

    async def test_raises_when_agent_name_not_registered(self) -> None:
        wf = ParallelWorkflow("test", agent_order=["nonexistent"])
        with pytest.raises(WorkflowExecutionError, match="not registered"):
            await wf.run()

    async def test_agent_order_respected(self) -> None:
        a = AppendAgent(make_profile("a"), "A")
        b = AppendAgent(make_profile("b"), "B")
        wf = ParallelWorkflow("test", agent_order=["a", "b"])
        wf.register_agent(a)
        wf.register_agent(b)
        final = await wf.run(State())
        ai_msgs = [m.content for m in final.messages if m.role == MessageRole.AI]
        assert set(ai_msgs) == {"response-A", "response-B"}
