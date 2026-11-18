"""Integration tests for SequentialWorkflow.

Tests a 2-agent pipeline (researcher → writer) using Agent backed by a
mocked litellm.acompletion — no live API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from miniagent.core.message import MessageRole
from miniagent.core.state import WorkflowStatus
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.model_config import ModelConfig
from miniagent.workflow.sequential import SequentialWorkflow
from tests.integration.conftest import (
    make_agent,
    make_litellm_response,
)


@pytest.fixture()
def researcher_profile() -> AgentProfile:
    return AgentProfile(
        name="researcher",
        model=ModelConfig(id="openai/gpt-4o"),
        system_prompt="You are a researcher.",
        max_iterations=3,
    )


@pytest.fixture()
def writer_profile() -> AgentProfile:
    return AgentProfile(
        name="writer",
        model=ModelConfig(id="openai/gpt-4o"),
        system_prompt="You are a writer.",
        max_iterations=3,
    )


class TestSequentialWorkflowIntegration:
    async def test_two_agent_pipeline_completes(
        self,
        base_state: object,
        mock_litellm: AsyncMock,
        researcher_profile: AgentProfile,
        writer_profile: AgentProfile,
    ) -> None:
        mock_litellm.side_effect = [
            make_litellm_response("Research findings here."),
            make_litellm_response("Polished article here."),
        ]

        researcher = make_agent(researcher_profile)
        writer = make_agent(writer_profile)

        wf = SequentialWorkflow(
            name="research-pipeline",
            agent_order=["researcher", "writer"],
        )
        wf.register_agent(researcher)
        wf.register_agent(writer)

        final = await wf.run(base_state)  # type: ignore[arg-type]

        assert final.status == WorkflowStatus.COMPLETED

    async def test_messages_from_both_agents_in_state(
        self,
        base_state: object,
        mock_litellm: AsyncMock,
        researcher_profile: AgentProfile,
        writer_profile: AgentProfile,
    ) -> None:
        mock_litellm.side_effect = [
            make_litellm_response("Research: topic A."),
            make_litellm_response("Article: topic A refined."),
        ]

        researcher = make_agent(researcher_profile)
        writer = make_agent(writer_profile)

        wf = SequentialWorkflow(
            name="pipeline",
            agent_order=["researcher", "writer"],
        )
        wf.register_agent(researcher)
        wf.register_agent(writer)

        final = await wf.run(base_state)  # type: ignore[arg-type]

        ai_messages = [m for m in final.messages if m.role == MessageRole.AI]
        assert len(ai_messages) == 2
        contents = [m.content for m in ai_messages]
        assert "Research: topic A." in contents
        assert "Article: topic A refined." in contents

    async def test_agent_messages_in_pipeline_order(
        self,
        base_state: object,
        mock_litellm: AsyncMock,
        researcher_profile: AgentProfile,
        writer_profile: AgentProfile,
    ) -> None:
        mock_litellm.side_effect = [
            make_litellm_response("First: researcher output."),
            make_litellm_response("Second: writer output."),
        ]

        researcher = make_agent(researcher_profile)
        writer = make_agent(writer_profile)

        wf = SequentialWorkflow(
            name="ordered-pipeline",
            agent_order=["researcher", "writer"],
        )
        wf.register_agent(researcher)
        wf.register_agent(writer)

        final = await wf.run(base_state)  # type: ignore[arg-type]

        ai_messages = [m for m in final.messages if m.role == MessageRole.AI]
        assert ai_messages[0].content == "First: researcher output."
        assert ai_messages[1].content == "Second: writer output."

    async def test_step_count_equals_number_of_agents(
        self,
        base_state: object,
        mock_litellm: AsyncMock,
        researcher_profile: AgentProfile,
        writer_profile: AgentProfile,
    ) -> None:
        mock_litellm.return_value = make_litellm_response("ok")

        researcher = make_agent(researcher_profile)
        writer = make_agent(writer_profile)

        wf = SequentialWorkflow(
            name="step-count-test",
            agent_order=["researcher", "writer"],
        )
        wf.register_agent(researcher)
        wf.register_agent(writer)

        final = await wf.run(base_state)  # type: ignore[arg-type]

        # Each agent produces 1 AI message = 1 step
        assert final.step_count == 2

    async def test_llm_called_once_per_agent(
        self,
        base_state: object,
        mock_litellm: AsyncMock,
        researcher_profile: AgentProfile,
        writer_profile: AgentProfile,
    ) -> None:
        mock_litellm.return_value = make_litellm_response("ok")

        researcher = make_agent(researcher_profile)
        writer = make_agent(writer_profile)

        wf = SequentialWorkflow(
            name="call-count-test",
            agent_order=["researcher", "writer"],
        )
        wf.register_agent(researcher)
        wf.register_agent(writer)

        await wf.run(base_state)  # type: ignore[arg-type]

        assert mock_litellm.call_count == 2
