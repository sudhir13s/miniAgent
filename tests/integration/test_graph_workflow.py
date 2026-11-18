"""Integration tests for GraphWorkflow.

Tests a writer → conditional-router → revise loop using mocked LLM calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from miniagent.core.message import MessageRole
from miniagent.core.state import State, WorkflowStatus
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.model_config import ModelConfig
from miniagent.workflow.graph import END_NODE, GraphWorkflow
from tests.integration.conftest import (
    make_agent,
    make_litellm_response,
)


@pytest.fixture()
def writer_profile() -> AgentProfile:
    return AgentProfile(
        name="writer",
        model=ModelConfig(id="openai/gpt-4o"),
        system_prompt="Write a draft.",
        max_iterations=3,
    )


@pytest.fixture()
def revise_profile() -> AgentProfile:
    return AgentProfile(
        name="reviser",
        model=ModelConfig(id="openai/gpt-4o"),
        system_prompt="Revise the draft.",
        max_iterations=3,
    )


def _needs_revision_router(state: State) -> str:
    """Route to 'reviser' if needs_revision is True, else END."""
    if state.variables.get("needs_revision", False):
        return "reviser"
    return END_NODE


class TestGraphWorkflowIntegration:
    async def test_single_pass_terminates_at_end(
        self,
        base_state: object,
        mock_litellm: AsyncMock,
        writer_profile: AgentProfile,
        revise_profile: AgentProfile,
    ) -> None:
        """When needs_revision is False after writer, graph should terminate."""
        mock_litellm.return_value = make_litellm_response("Draft complete.")

        writer = make_agent(writer_profile)
        reviser = make_agent(revise_profile)

        wf = GraphWorkflow(name="draft-flow", entry_point="writer")
        wf.register_agent(writer)
        wf.register_agent(reviser)
        wf.add_conditional_edge("writer", _needs_revision_router)
        wf.add_edge("reviser", END_NODE)

        # needs_revision not set → router returns END_NODE
        final = await wf.run(base_state)  # type: ignore[arg-type]

        assert final.status == WorkflowStatus.COMPLETED
        ai_msgs = [m for m in final.messages if m.role == MessageRole.AI]
        assert len(ai_msgs) == 1

    async def test_revision_loop_runs_once_then_exits(
        self,
        base_state: Any,
        mock_litellm: AsyncMock,
        writer_profile: AgentProfile,
        revise_profile: AgentProfile,
    ) -> None:
        """Writer sets needs_revision=True; reviser clears it → 1 revision."""
        call_count = 0

        async def _llm_side_effect(*_args: Any, **_kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return make_litellm_response(f"Response {call_count}")

        mock_litellm.side_effect = _llm_side_effect

        # Patch writer agent.run to also set needs_revision=True then False
        revision_toggle = {"count": 0}

        writer_agent = make_agent(writer_profile)
        reviser_agent = make_agent(revise_profile)

        # Override writer run: sets needs_revision=True first call
        original_writer_run = writer_agent.run

        async def _writer_run(state: State) -> State:
            result = await original_writer_run(state)
            revision_toggle["count"] += 1
            return result.model_copy(
                update={"variables": {**dict(result.variables), "needs_revision": True}}
            )

        writer_agent.run = _writer_run  # type: ignore[method-assign]

        # Override reviser run: clears needs_revision
        original_reviser_run = reviser_agent.run

        async def _reviser_run(state: State) -> State:
            result = await original_reviser_run(state)
            return result.model_copy(
                update={"variables": {**dict(result.variables), "needs_revision": False}}
            )

        reviser_agent.run = _reviser_run  # type: ignore[method-assign]

        wf = GraphWorkflow(name="revision-flow", entry_point="writer")
        wf.register_agent(writer_agent)
        wf.register_agent(reviser_agent)
        wf.add_conditional_edge("writer", _needs_revision_router)
        wf.add_conditional_edge("reviser", _needs_revision_router)

        final = await wf.run(base_state)

        assert final.status == WorkflowStatus.COMPLETED
        assert not final.variables.get("needs_revision")
        ai_msgs = [m for m in final.messages if m.role == MessageRole.AI]
        # writer called once, reviser called once
        assert len(ai_msgs) == 2

    async def test_max_steps_stops_infinite_loop(
        self,
        mock_litellm: AsyncMock,
        writer_profile: AgentProfile,
    ) -> None:
        """A graph that always routes back to itself must stop at max_steps."""
        mock_litellm.return_value = make_litellm_response("looping")

        writer = make_agent(writer_profile)
        wf = GraphWorkflow(name="loop-stopper", entry_point="writer")
        wf.register_agent(writer)
        # Always route back to writer
        wf.add_conditional_edge("writer", lambda _state: "writer")

        state = State(
            messages=[],
            max_steps=3,
        )
        from miniagent.core.exceptions import WorkflowExecutionError

        with pytest.raises(WorkflowExecutionError, match="max steps"):
            await wf.run(state)

    async def test_deterministic_edge_terminates(
        self,
        base_state: object,
        mock_litellm: AsyncMock,
        writer_profile: AgentProfile,
        revise_profile: AgentProfile,
    ) -> None:
        """A fixed edge writer → reviser → END_NODE should complete in 2 steps."""
        mock_litellm.return_value = make_litellm_response("ok")

        writer = make_agent(writer_profile)
        reviser = make_agent(revise_profile)

        wf = GraphWorkflow(name="linear-graph", entry_point="writer")
        wf.register_agent(writer)
        wf.register_agent(reviser)
        wf.add_edge("writer", "reviser")
        wf.add_edge("reviser", END_NODE)

        final = await wf.run(base_state)  # type: ignore[arg-type]

        assert final.status == WorkflowStatus.COMPLETED
        ai_msgs = [m for m in final.messages if m.role == MessageRole.AI]
        assert len(ai_msgs) == 2
