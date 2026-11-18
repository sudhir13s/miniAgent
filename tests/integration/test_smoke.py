"""Integration smoke tests — no external API calls required.

Verifies that core classes can be imported, instantiated, and composed
without any live LLM dependencies.
"""

from __future__ import annotations

import pytest

from miniagent.agent import Agent, build_agent
from miniagent.core.exceptions import ConfigurationError
from miniagent.core.message import Message, MessageRole
from miniagent.core.state import State, WorkflowStatus
from miniagent.llm.registry import get_default_provider
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.model_config import ModelConfig
from miniagent.workflow.sequential import SequentialWorkflow


def _make_profile(name: str = "smoke-agent", tools: list[str] | None = None) -> AgentProfile:
    return AgentProfile(
        name=name,
        model=ModelConfig(id="openai/gpt-4o"),
        tools=tools or [],
    )


def _make_agent(name: str = "smoke-agent") -> Agent:
    return Agent(_make_profile(name), get_default_provider())


class TestCoreImports:
    """Core classes import and instantiate without errors."""

    def test_agent_profile_instantiates(self) -> None:
        profile = _make_profile()
        assert profile.name == "smoke-agent"
        assert profile.model.id == "openai/gpt-4o"

    def test_state_instantiates(self) -> None:
        state = State()
        assert state.status == WorkflowStatus.PENDING
        assert state.step_count == 0

    def test_message_instantiates(self) -> None:
        msg = Message(role=MessageRole.USER, content="hello")
        assert msg.content == "hello"


class TestAgentConstruction:
    """Agents are built directly or from a profile via build_agent."""

    def test_instantiate_agent_directly(self) -> None:
        """Agent is constructed directly (requires llm_provider injection)."""
        agent = _make_agent()
        assert isinstance(agent, Agent)
        assert agent.profile.name == "smoke-agent"

    def test_build_agent_attaches_listed_builtin_tools(self) -> None:
        agent = build_agent(_make_profile(tools=["calculator", "timestamp"]))
        assert isinstance(agent, Agent)
        assert sorted(agent.tools) == ["calculator", "timestamp"]
        assert agent.llm_provider is get_default_provider()

    def test_build_agent_raises_for_unknown_tool(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown tools: no_such_tool"):
            build_agent(_make_profile(tools=["no_such_tool"]))


class TestConfigLoading:
    """AgentProfile.from_dict builds profiles from a config mapping."""

    def test_from_dict(self) -> None:
        profiles = AgentProfile.from_dict(
            {
                "agents": [
                    {
                        "name": "test-agent",
                        "model": {"id": "openai/gpt-4o"},
                    }
                ]
            }
        )
        assert len(profiles) == 1
        assert profiles[0].name == "test-agent"
        assert profiles[0].model.id == "openai/gpt-4o"


class TestSequentialWorkflow:
    """SequentialWorkflow composes without executing LLM calls."""

    def test_register_agents(self) -> None:
        wf = SequentialWorkflow(name="smoke-flow", agent_order=["a", "b"])
        assert wf.name == "smoke-flow"
        assert wf.agent_order == ["a", "b"]
        assert wf.agents == {}

    def test_duplicate_agent_raises(self) -> None:
        wf = SequentialWorkflow(name="smoke-flow", agent_order=["a"])
        agent = _make_agent("a")
        wf.register_agent(agent)
        with pytest.raises(ValueError, match="already registered"):
            wf.register_agent(agent)
