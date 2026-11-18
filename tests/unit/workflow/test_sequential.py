"""Tests for SequentialWorkflow."""

from __future__ import annotations

import pytest

from miniagent.core.exceptions import WorkflowExecutionError
from miniagent.core.state import State, WorkflowStatus
from miniagent.workflow.sequential import SequentialWorkflow
from tests.unit.workflow.fakes import (
    EchoAgent,
    FailingAgent,
    make_profile,
)


class TestSequentialWorkflow:
    """Tests for SequentialWorkflow."""

    @pytest.mark.asyncio
    async def test_sequential_happy_path(self) -> None:
        """Should run agents in order."""
        wf = SequentialWorkflow(name="test-seq")
        agent_a = EchoAgent(make_profile("agent-a"))
        agent_b = EchoAgent(make_profile("agent-b"))
        wf.register_agent(agent_a)
        wf.register_agent(agent_b)

        result = await wf.run()
        assert result.status == WorkflowStatus.COMPLETED
        assert result.step_count == 2
        assert len(result.messages) == 2
        assert "agent-a" in result.messages[0].content
        assert "agent-b" in result.messages[1].content

    @pytest.mark.asyncio
    async def test_sequential_custom_order(self) -> None:
        """Should respect agent_order."""
        wf = SequentialWorkflow(name="test-seq", agent_order=["agent-b", "agent-a"])
        wf.register_agent(EchoAgent(make_profile("agent-a")))
        wf.register_agent(EchoAgent(make_profile("agent-b")))

        result = await wf.run()
        assert "agent-b" in result.messages[0].content
        assert "agent-a" in result.messages[1].content

    @pytest.mark.asyncio
    async def test_sequential_no_agents_raises(self) -> None:
        """Should raise when no agents are registered."""
        wf = SequentialWorkflow(name="empty")
        with pytest.raises(WorkflowExecutionError, match="no agents"):
            await wf.run()

    @pytest.mark.asyncio
    async def test_sequential_missing_agent_raises(self) -> None:
        """Should raise when agent_order references unregistered agent."""
        wf = SequentialWorkflow(name="test", agent_order=["nonexistent"])
        with pytest.raises(WorkflowExecutionError, match="not registered"):
            await wf.run()

    @pytest.mark.asyncio
    async def test_sequential_agent_failure(self) -> None:
        """Should propagate agent failure as WorkflowExecutionError."""
        wf = SequentialWorkflow(name="failing")
        wf.register_agent(FailingAgent(make_profile("bad-agent")))
        with pytest.raises(WorkflowExecutionError, match="failed"):
            await wf.run()

    @pytest.mark.asyncio
    async def test_sequential_with_initial_state(self) -> None:
        """Should accept an initial state."""
        wf = SequentialWorkflow(name="test")
        wf.register_agent(EchoAgent(make_profile("agent-a")))
        initial = State(variables={"key": "value"})
        result = await wf.run(state=initial)
        assert result.variables["key"] == "value"
