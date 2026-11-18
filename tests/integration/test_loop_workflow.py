"""Integration tests for LoopWorkflow.

Tests iterative execution with exit conditions and max_iterations guard.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from miniagent.core.message import MessageRole
from miniagent.core.state import State, WorkflowStatus
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.model_config import ModelConfig
from miniagent.workflow.loop import LoopWorkflow
from tests.integration.conftest import (
    make_agent,
    make_litellm_response,
)


@pytest.fixture()
def refiner_profile() -> AgentProfile:
    return AgentProfile(
        name="refiner",
        model=ModelConfig(id="openai/gpt-4o"),
        system_prompt="Refine the content.",
        max_iterations=3,
    )


class TestLoopWorkflowIntegration:
    async def test_exits_when_exit_condition_met(
        self,
        base_state: State,
        mock_litellm: AsyncMock,
        refiner_profile: AgentProfile,
    ) -> None:
        """Loop exits on iteration 3 when quality_score reaches 3."""
        iteration = {"count": 0}

        refiner = make_agent(refiner_profile)
        original_run = refiner.run

        async def _run(state: State) -> State:
            result = await original_run(state)
            iteration["count"] += 1
            return result.model_copy(
                update={
                    "variables": {
                        **dict(result.variables),
                        "quality_score": iteration["count"],
                    }
                }
            )

        refiner.run = _run  # type: ignore[method-assign]
        mock_litellm.return_value = make_litellm_response("iteration output")

        def _exit_at_3(state: State) -> bool:
            return state.variables.get("quality_score", 0) >= 3

        wf = LoopWorkflow(
            name="quality-loop",
            agent_order=["refiner"],
            exit_condition=_exit_at_3,
            max_iterations=10,
        )
        wf.register_agent(refiner)

        final = await wf.run(base_state)

        assert final.status == WorkflowStatus.COMPLETED
        assert final.variables["quality_score"] == 3
        assert iteration["count"] == 3

    async def test_max_iterations_guard_stops_loop(
        self,
        base_state: State,
        mock_litellm: AsyncMock,
        refiner_profile: AgentProfile,
    ) -> None:
        """When exit condition is never met, loop stops at max_iterations."""
        mock_litellm.return_value = make_litellm_response("never enough")

        # Exit condition that never triggers
        def _never_exit(_state: State) -> bool:
            return False

        wf = LoopWorkflow(
            name="bounded-loop",
            agent_order=["refiner"],
            exit_condition=_never_exit,
            max_iterations=3,
        )
        wf.register_agent(make_agent(refiner_profile))

        final = await wf.run(base_state)

        assert final.status == WorkflowStatus.COMPLETED
        # 3 iterations x 1 AI message per iteration = 3 AI messages
        ai_msgs = [m for m in final.messages if m.role == MessageRole.AI]
        assert len(ai_msgs) == 3

    async def test_llm_called_correct_number_of_times(
        self,
        base_state: State,
        mock_litellm: AsyncMock,
        refiner_profile: AgentProfile,
    ) -> None:
        """Loop with max_iterations=4, no exit condition → 4 LLM calls."""
        mock_litellm.return_value = make_litellm_response("ok")

        wf = LoopWorkflow(
            name="call-count-loop",
            agent_order=["refiner"],
            max_iterations=4,
        )
        wf.register_agent(make_agent(refiner_profile))

        await wf.run(base_state)

        assert mock_litellm.call_count == 4

    async def test_loop_with_no_exit_condition_runs_all_iterations(
        self,
        base_state: State,
        mock_litellm: AsyncMock,
        refiner_profile: AgentProfile,
    ) -> None:
        """No exit_condition → runs exactly max_iterations times."""
        mock_litellm.return_value = make_litellm_response("step")

        wf = LoopWorkflow(
            name="no-exit-loop",
            agent_order=["refiner"],
            max_iterations=5,
        )
        wf.register_agent(make_agent(refiner_profile))

        final = await wf.run(base_state)

        assert final.status == WorkflowStatus.COMPLETED
        assert mock_litellm.call_count == 5

    async def test_exits_immediately_if_condition_true_after_first_iteration(
        self,
        base_state: State,
        mock_litellm: AsyncMock,
        refiner_profile: AgentProfile,
    ) -> None:
        mock_litellm.return_value = make_litellm_response("done")

        refiner = make_agent(refiner_profile)
        original = refiner.run

        async def _run(state: State) -> State:
            result = await original(state)
            return result.model_copy(update={"variables": {**dict(result.variables), "done": True}})

        refiner.run = _run  # type: ignore[method-assign]

        wf = LoopWorkflow(
            name="immediate-exit",
            agent_order=["refiner"],
            exit_condition=lambda s: s.variables.get("done", False),
            max_iterations=10,
        )
        wf.register_agent(refiner)

        final = await wf.run(base_state)

        assert final.status == WorkflowStatus.COMPLETED
        # Should have exited after just 1 iteration
        assert mock_litellm.call_count == 1
