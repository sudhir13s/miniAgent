"""Tests for AgentProfile: validation, and loading with from_yaml / from_dict."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from miniagent.core.exceptions import ConfigurationError
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.model_config import ModelConfig


class TestAgentProfile:
    """Tests for the AgentProfile model."""

    def _make_model_config(self) -> ModelConfig:
        return ModelConfig(id="gemini/gemini-1.5-pro")

    def test_create_profile(self) -> None:
        """A valid agent profile should be created."""
        profile = AgentProfile(
            name="researcher",
            model=self._make_model_config(),
        )
        assert profile.name == "researcher"
        assert profile.model.id == "gemini/gemini-1.5-pro"
        assert profile.system_prompt == "You are a helpful AI assistant."
        assert profile.tools == []

    def test_invalid_name_format(self) -> None:
        """Names with invalid characters should be rejected."""
        with pytest.raises(ValidationError, match="must contain only"):
            AgentProfile(
                name="invalid name!",
                model=self._make_model_config(),
            )

    def test_hyphen_underscore_name_allowed(self) -> None:
        """Names with hyphens and underscores should be allowed."""
        profile = AgentProfile(
            name="code-reviewer_v2",
            model=self._make_model_config(),
        )
        assert profile.name == "code-reviewer_v2"

    def test_empty_name_rejected(self) -> None:
        """Empty name should be rejected."""
        with pytest.raises(ValidationError):
            AgentProfile(name="", model=self._make_model_config())

    def test_max_iterations_bounds(self) -> None:
        """max_iterations must be between 1 and 100."""
        with pytest.raises(ValidationError):
            AgentProfile(
                name="test",
                model=self._make_model_config(),
                max_iterations=0,
            )
        with pytest.raises(ValidationError):
            AgentProfile(
                name="test",
                model=self._make_model_config(),
                max_iterations=101,
            )

    def test_profile_with_tools(self) -> None:
        """An agent profile should accept a tools list."""
        profile = AgentProfile(
            name="agent",
            model=self._make_model_config(),
            tools=["search", "calculator"],
        )
        assert profile.tools == ["search", "calculator"]

    def test_frozen(self) -> None:
        """AgentProfile should be immutable."""
        profile = AgentProfile(
            name="agent",
            model=self._make_model_config(),
        )
        with pytest.raises(ValidationError):
            profile.name = "other"  # type: ignore[misc]


_TWO_AGENTS = textwrap.dedent("""\
    agents:
      - name: researcher
        description: "Research agent"
        system_prompt: "You are a researcher."
        model:
          id: gemini/gemini-1.5-pro
          temperature: 0.3
        tools: [web_fetch]
        max_iterations: 15
      - name: writer
        model:
          id: openai/gpt-4o
""")


@pytest.fixture()
def two_agents_file(tmp_path: Path) -> Path:
    path = tmp_path / "agents.yaml"
    path.write_text(_TWO_AGENTS, encoding="utf-8")
    return path


def _config(*agents: dict[str, Any]) -> dict[str, Any]:
    return {"agents": list(agents)}


def _agent(name: str, **fields: Any) -> dict[str, Any]:
    return {"name": name, "model": {"id": "openai/gpt-4o"}, **fields}


class TestFromYaml:
    def test_should_load_every_agent_in_file_order(self, two_agents_file: Path) -> None:
        profiles = AgentProfile.from_yaml(two_agents_file)
        assert [profile.name for profile in profiles] == ["researcher", "writer"]
        assert profiles[0].model.temperature == 0.3
        assert profiles[0].tools == ["web_fetch"]

    def test_should_return_one_agent_by_name(self, two_agents_file: Path) -> None:
        writer = AgentProfile.from_yaml(two_agents_file, name="writer")
        assert isinstance(writer, AgentProfile)
        assert writer.model.id == "openai/gpt-4o"

    def test_should_list_defined_agents_when_the_name_is_unknown(
        self, two_agents_file: Path
    ) -> None:
        with pytest.raises(ConfigurationError, match="Defined agents: researcher, writer"):
            AgentProfile.from_yaml(two_agents_file, name="editor")

    def test_should_report_file_problems(self) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            AgentProfile.from_yaml("/nonexistent/agents.yaml")


class TestFromDict:
    def test_should_load_agents_from_a_config_mapping(self) -> None:
        profiles = AgentProfile.from_dict(_config(_agent("a"), _agent("b")))
        assert [profile.name for profile in profiles] == ["a", "b"]

    def test_should_return_one_agent_by_name(self) -> None:
        profile = AgentProfile.from_dict(_config(_agent("a"), _agent("b")), name="b")
        assert profile.name == "b"

    def test_should_point_single_agent_fields_to_the_constructor(self) -> None:
        with pytest.raises(
            ConfigurationError, match=r"Missing 'agents' key.*AgentProfile\(\.\.\.\)"
        ):
            AgentProfile.from_dict(_agent("a"))

    def test_should_reject_duplicate_names(self) -> None:
        with pytest.raises(ConfigurationError, match="Duplicate agent name 'a'"):
            AgentProfile.from_dict(_config(_agent("a"), _agent("a")))

    def test_should_reject_unknown_agent_keys(self) -> None:
        misspelled = _agent("a", max_iteration=3)
        with pytest.raises(ConfigurationError, match="max_iteration"):
            AgentProfile.from_dict(_config(misspelled))

    def test_should_report_which_agent_failed_validation(self) -> None:
        bad = _agent("b", model={"id": "openai/gpt-4o", "temperature": 5.0})
        with pytest.raises(ConfigurationError, match="Validation failed for agent at index 1"):
            AgentProfile.from_dict(_config(_agent("a"), bad))
