"""Unit tests for API dependency injection helpers.

Route tests are covered comprehensively in test_routes.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from miniagent.agent.agent import Agent
from miniagent.api.dependencies import (
    create_session_store,
    get_agent_by_name,
    get_agent_profiles,
    reset_profile_cache,
)
from miniagent.core.exceptions import ConfigurationError, SessionStoreError
from miniagent.memory.session_store import InMemorySessionStore
from miniagent.memory.sqlite_session_store import SQLiteSessionStore


class TestCreateSessionStore:
    def test_should_keep_sessions_in_memory_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SESSION_DB_PATH", raising=False)
        assert isinstance(create_session_store(), InMemorySessionStore)

    def test_should_use_sqlite_when_a_database_path_is_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        database = tmp_path / "sessions.db"
        monkeypatch.setenv("SESSION_DB_PATH", str(database))
        store = create_session_store()
        assert isinstance(store, SQLiteSessionStore)
        assert store.path == database

    def test_should_fail_fast_when_the_database_cannot_be_opened(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "no-such-dir" / "sessions.db"))
        with pytest.raises(SessionStoreError):
            create_session_store()


class TestGetAgentProfiles:
    """Tests for the get_agent_profiles dependency."""

    def setup_method(self) -> None:
        reset_profile_cache()

    def teardown_method(self) -> None:
        reset_profile_cache()

    def test_loads_profiles_from_path(self, tmp_path: Path) -> None:
        """get_agent_profiles loads YAML from the given path."""
        yaml_content = """
agents:
  - name: test-agent
    model:
      id: openai/gpt-4o
"""
        config = tmp_path / "agents.yaml"
        config.write_text(yaml_content)

        profiles = get_agent_profiles(str(config))
        assert len(profiles) == 1
        assert profiles[0].name == "test-agent"

    def test_raises_configuration_error_on_bad_file(self) -> None:
        """get_agent_profiles raises ConfigurationError for missing file."""
        reset_profile_cache()
        with pytest.raises(ConfigurationError):
            get_agent_profiles("/nonexistent/path/agents.yaml")

    def test_env_var_used_when_no_path_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_agent_profiles reads AGENTS_CONFIG_PATH env var."""
        yaml_content = """
agents:
  - name: env-agent
    model:
      id: openai/gpt-4o
"""
        config = tmp_path / "agents.yaml"
        config.write_text(yaml_content)
        monkeypatch.setenv("AGENTS_CONFIG_PATH", str(config))
        reset_profile_cache()

        profiles = get_agent_profiles()
        assert profiles[0].name == "env-agent"


class TestGetAgentByName:
    """Tests for the get_agent_by_name dependency."""

    def setup_method(self) -> None:
        reset_profile_cache()

    def teardown_method(self) -> None:
        reset_profile_cache()

    def test_returns_agent_for_known_name(self, tmp_path: Path) -> None:
        """get_agent_by_name returns an agent when the name matches."""
        yaml_content = """
agents:
  - name: my-agent
    model:
      id: openai/gpt-4o
"""
        config = tmp_path / "agents.yaml"
        config.write_text(yaml_content)

        agent = get_agent_by_name("my-agent", str(config))

        assert isinstance(agent, Agent)
        assert agent.profile.name == "my-agent"

    def test_raises_for_unknown_name(self, tmp_path: Path) -> None:
        """get_agent_by_name raises ConfigurationError for unknown agent."""
        yaml_content = """
agents:
  - name: only-agent
    model:
      id: openai/gpt-4o
"""
        config = tmp_path / "agents.yaml"
        config.write_text(yaml_content)

        with pytest.raises(ConfigurationError, match="not found"):
            get_agent_by_name("nonexistent", str(config))


class TestResetProfileCache:
    """Tests for reset_profile_cache."""

    def test_reset_clears_cache(self, tmp_path: Path) -> None:
        """reset_profile_cache allows reloading profiles."""
        yaml_content = """
agents:
  - name: cached-agent
    model:
      id: openai/gpt-4o
"""
        config = tmp_path / "agents.yaml"
        config.write_text(yaml_content)

        get_agent_profiles(str(config))
        reset_profile_cache()
        profiles = get_agent_profiles(str(config))
        assert profiles[0].name == "cached-agent"
