"""Integration tests for ParallelWorkflow.

Tests 3 agents running concurrently, result merging, and variable precedence.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from miniagent.core.message import MessageRole
from miniagent.core.state import State, WorkflowStatus
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.model_config import ModelConfig
from miniagent.workflow.parallel import ParallelWorkflow
from tests.integration.conftest import (
    make_agent,
    make_litellm_response,
)


@pytest.fixture()
def agent_a_profile() -> AgentProfile:
    return AgentProfile(
        name="agent-a",
        model=ModelConfig(id="openai/gpt-4o"),
        system_prompt="You are agent A.",
        max_iterations=3,
    )


@pytest.fixture()
def agent_b_profile() -> AgentProfile:
    return AgentProfile(
        name="agent-b",
        model=ModelConfig(id="openai/gpt-4o"),
        system_prompt="You are agent B.",
        max_iterations=3,
    )


@pytest.fixture()
def agent_c_profile() -> AgentProfile:
    return AgentProfile(
        name="agent-c",
        model=ModelConfig(id="openai/gpt-4o"),
        system_prompt="You are agent C.",
        max_iterations=3,
    )


class TestParallelWorkflowIntegration:
    async def test_all_three_agents_produce_messages(
        self,
        base_state: object,
        mock_litellm: AsyncMock,
        agent_a_profile: AgentProfile,
        agent_b_profile: AgentProfile,
        agent_c_profile: AgentProfile,
    ) -> None:
        mock_litellm.return_value = make_litellm_response("parallel response")

        agent_a = make_agent(agent_a_profile)
        agent_b = make_agent(agent_b_profile)
        agent_c = make_agent(agent_c_profile)

        wf = ParallelWorkflow(
            name="fan-out",
            agent_order=["agent-a", "agent-b", "agent-c"],
        )
        wf.register_agent(agent_a)
        wf.register_agent(agent_b)
        wf.register_agent(agent_c)

        final = await wf.run(base_state)  # type: ignore[arg-type]

        ai_msgs = [m for m in final.messages if m.role == MessageRole.AI]
        assert len(ai_msgs) == 3

    async def test_workflow_status_is_completed(
        self,
        base_state: object,
        mock_litellm: AsyncMock,
        agent_a_profile: AgentProfile,
        agent_b_profile: AgentProfile,
    ) -> None:
        mock_litellm.return_value = make_litellm_response("ok")

        wf = ParallelWorkflow(
            name="status-test",
            agent_order=["agent-a", "agent-b"],
        )
        wf.register_agent(make_agent(agent_a_profile))
        wf.register_agent(make_agent(agent_b_profile))

        final = await wf.run(base_state)  # type: ignore[arg-type]

        assert final.status == WorkflowStatus.COMPLETED

    async def test_variable_merge_last_write_wins(
        self,
        base_state: State,
        mock_litellm: AsyncMock,
        agent_a_profile: AgentProfile,
        agent_b_profile: AgentProfile,
    ) -> None:
        """agent-b is later in agent_order, so its variable wins on collision."""
        mock_litellm.return_value = make_litellm_response("ok")

        # Override run to set a variable in each agent
        agent_a = make_agent(agent_a_profile)
        agent_b = make_agent(agent_b_profile)

        original_a = agent_a.run
        original_b = agent_b.run

        async def _a_run(state: State) -> State:
            result = await original_a(state)
            return result.model_copy(
                update={"variables": {**dict(result.variables), "result": "from-a"}}
            )

        async def _b_run(state: State) -> State:
            result = await original_b(state)
            return result.model_copy(
                update={"variables": {**dict(result.variables), "result": "from-b"}}
            )

        agent_a.run = _a_run  # type: ignore[method-assign]
        agent_b.run = _b_run  # type: ignore[method-assign]

        wf = ParallelWorkflow(
            name="merge-test",
            agent_order=["agent-a", "agent-b"],
        )
        wf.register_agent(agent_a)
        wf.register_agent(agent_b)

        final = await wf.run(base_state)

        # Last in agent_order wins
        assert final.variables["result"] == "from-b"

    async def test_step_count_is_sum_of_branches(
        self,
        base_state: object,
        mock_litellm: AsyncMock,
        agent_a_profile: AgentProfile,
        agent_b_profile: AgentProfile,
        agent_c_profile: AgentProfile,
    ) -> None:
        mock_litellm.return_value = make_litellm_response("ok")

        wf = ParallelWorkflow(
            name="step-sum",
            agent_order=["agent-a", "agent-b", "agent-c"],
        )
        wf.register_agent(make_agent(agent_a_profile))
        wf.register_agent(make_agent(agent_b_profile))
        wf.register_agent(make_agent(agent_c_profile))

        final = await wf.run(base_state)  # type: ignore[arg-type]

        # Each agent produces 1 step, 3 branches = step_count of 3
        assert final.step_count == 3

    async def test_llm_called_once_per_concurrent_agent(
        self,
        base_state: object,
        mock_litellm: AsyncMock,
        agent_a_profile: AgentProfile,
        agent_b_profile: AgentProfile,
        agent_c_profile: AgentProfile,
    ) -> None:
        mock_litellm.return_value = make_litellm_response("ok")

        wf = ParallelWorkflow(
            name="call-count",
            agent_order=["agent-a", "agent-b", "agent-c"],
        )
        wf.register_agent(make_agent(agent_a_profile))
        wf.register_agent(make_agent(agent_b_profile))
        wf.register_agent(make_agent(agent_c_profile))

        await wf.run(base_state)  # type: ignore[arg-type]

        assert mock_litellm.call_count == 3
