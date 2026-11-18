"""Tests for GraphWorkflow routing."""

from __future__ import annotations

import pytest

from miniagent.core.exceptions import WorkflowExecutionError
from miniagent.core.state import State, WorkflowStatus
from miniagent.workflow.graph import END_NODE, GraphWorkflow
from tests.unit.workflow.fakes import (
    EchoAgent,
    FailingAgent,
    make_profile,
)


class TestGraphWorkflow:
    """Tests for GraphWorkflow."""

    @pytest.mark.asyncio
    async def test_graph_linear_path(self) -> None:
        """Should follow deterministic edges."""
        wf = GraphWorkflow(name="linear", entry_point="a")
        wf.register_agent(EchoAgent(make_profile("a")))
        wf.register_agent(EchoAgent(make_profile("b")))
        wf.add_edge("a", "b")
        wf.add_edge("b", END_NODE)

        result = await wf.run()
        assert result.status == WorkflowStatus.COMPLETED
        assert result.step_count == 2

    @pytest.mark.asyncio
    async def test_graph_conditional_edge(self) -> None:
        """Should follow conditional routing."""
        wf = GraphWorkflow(name="cond", entry_point="a")
        wf.register_agent(EchoAgent(make_profile("a")))
        wf.register_agent(EchoAgent(make_profile("b")))
        wf.register_agent(EchoAgent(make_profile("c")))

        def router(state: State) -> str:
            return "c" if state.step_count > 0 else "b"

        wf.add_conditional_edge("a", router)
        wf.add_edge("b", END_NODE)
        wf.add_edge("c", END_NODE)

        result = await wf.run()
        # After 'a' executes, step_count=1 > 0, so routes to 'c'
        assert result.step_count == 2
        assert "c" in result.messages[1].content

    @pytest.mark.asyncio
    async def test_graph_no_entry_point_raises(self) -> None:
        """Should raise without entry_point."""
        wf = GraphWorkflow(name="no-entry")
        with pytest.raises(WorkflowExecutionError, match="no entry point"):
            await wf.run()

    @pytest.mark.asyncio
    async def test_graph_unknown_entry_point_raises(self) -> None:
        """Should raise when entry_point is not registered."""
        wf = GraphWorkflow(name="bad-entry", entry_point="missing")
        with pytest.raises(WorkflowExecutionError, match="not a registered"):
            await wf.run()

    @pytest.mark.asyncio
    async def test_graph_max_steps_enforced(self) -> None:
        """Should raise when max steps reached (loop protection)."""
        wf = GraphWorkflow(name="loop", entry_point="a")
        wf.register_agent(EchoAgent(make_profile("a")))
        # Create a loop: a -> a
        wf.add_conditional_edge("a", lambda _: "a")

        state = State(max_steps=3)
        with pytest.raises(WorkflowExecutionError, match="exceeded max steps"):
            await wf.run(state=state)

    @pytest.mark.asyncio
    async def test_graph_node_with_no_edge_stops(self) -> None:
        """Should stop gracefully when no outgoing edge."""
        wf = GraphWorkflow(name="terminal", entry_point="a")
        wf.register_agent(EchoAgent(make_profile("a")))
        # No edge from 'a' — should terminate
        result = await wf.run()
        assert result.status == WorkflowStatus.COMPLETED
        assert result.step_count == 1

    @pytest.mark.asyncio
    async def test_graph_agent_failure(self) -> None:
        """Should propagate agent failure."""
        wf = GraphWorkflow(name="fail", entry_point="bad")
        wf.register_agent(FailingAgent(make_profile("bad")))
        with pytest.raises(WorkflowExecutionError, match="failed"):
            await wf.run()

    def test_duplicate_edge_raises(self) -> None:
        """Should reject duplicate outgoing edges."""
        wf = GraphWorkflow(name="dup")
        wf.add_edge("a", "b")
        with pytest.raises(ValueError, match="already exists"):
            wf.add_edge("a", "c")

    def test_duplicate_conditional_edge_raises(self) -> None:
        """Should reject duplicate conditional edges."""
        wf = GraphWorkflow(name="dup")
        wf.add_conditional_edge("a", lambda _: "b")
        with pytest.raises(ValueError, match="already exists"):
            wf.add_conditional_edge("a", lambda _: "c")
