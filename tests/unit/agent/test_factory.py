"""Tests for ``build_agent``: resolving a profile's tools and toolsets."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from miniagent.agent import build_agent
from miniagent.core.exceptions import ConfigurationError
from miniagent.memory.context_window import TokenBudgetPolicy
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.model_config import ModelConfig
from tests.unit.tool.fakes import FakeToolset, StaticTool


def _profile(*, tools: list[str] | None = None, toolsets: list[str] | None = None) -> AgentProfile:
    return AgentProfile(
        name="builder-test",
        model=ModelConfig(id="openai/gpt-4o"),
        tools=tools or [],
        toolsets=toolsets or [],
    )


class TestBuildAgentTools:
    def test_attaches_listed_builtin_tools(self) -> None:
        agent = build_agent(_profile(tools=["calculator"]), llm_provider=MagicMock())
        assert list(agent.tools) == ["calculator"]

    def test_custom_tool_overrides_builtin_with_same_name(self) -> None:
        custom = StaticTool("calculator")
        agent = build_agent(
            _profile(tools=["calculator"]),
            tools={"calculator": custom},
            llm_provider=MagicMock(),
        )
        assert agent.tools["calculator"] is custom

    def test_unknown_tool_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown tools: nope"):
            build_agent(_profile(tools=["nope"]), llm_provider=MagicMock())


class TestBuildAgentToolsets:
    def test_attaches_only_the_toolsets_the_profile_lists(self) -> None:
        docs = FakeToolset("docs", ["search"])
        github = FakeToolset("github", ["create_issue"])

        agent = build_agent(
            _profile(toolsets=["docs"]), toolsets=[docs, github], llm_provider=MagicMock()
        )

        assert agent.toolsets == [docs]

    def test_unknown_toolset_is_rejected_with_available_names(self) -> None:
        with pytest.raises(ConfigurationError, match=r"unknown toolsets: jira.*Available.*docs"):
            build_agent(
                _profile(toolsets=["jira"]),
                toolsets=[FakeToolset("docs", [])],
                llm_provider=MagicMock(),
            )

    def test_duplicate_toolset_names_are_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="more than one toolset named 'docs'"):
            build_agent(
                _profile(toolsets=["docs"]),
                toolsets=[FakeToolset("docs", []), FakeToolset("docs", [])],
                llm_provider=MagicMock(),
            )


class TestBuildAgentContextPolicy:
    def test_budget_on_profile_becomes_token_budget_policy(self) -> None:
        profile = _profile().model_copy(update={"max_context_tokens": 8000})
        agent = build_agent(profile, llm_provider=MagicMock())
        assert isinstance(agent.context_policy, TokenBudgetPolicy)
        assert agent.context_policy.max_tokens == 8000

    def test_no_budget_means_no_trimming(self) -> None:
        agent = build_agent(_profile(), llm_provider=MagicMock())
        assert agent.context_policy is None


class TestProfileContextBudgetField:
    def test_defaults_to_none(self) -> None:
        assert _profile().max_context_tokens is None

    def test_loads_from_yaml_mapping(self) -> None:
        profile = AgentProfile.model_validate(
            {"name": "budgeted", "model": {"id": "openai/gpt-4o"}, "max_context_tokens": 4096}
        )
        assert profile.max_context_tokens == 4096

    @pytest.mark.parametrize("budget", [0, -1])
    def test_must_be_positive(self, budget: int) -> None:
        with pytest.raises(ValueError, match="max_context_tokens"):
            AgentProfile(name="x", model=ModelConfig(id="openai/gpt-4o"), max_context_tokens=budget)


class TestProfileToolsetsField:
    def test_defaults_to_empty(self) -> None:
        assert _profile().toolsets == []

    def test_names_must_be_identifiers(self) -> None:
        with pytest.raises(ValueError, match="Toolset name"):
            _profile(toolsets=["bad name"])
