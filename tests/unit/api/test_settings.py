"""Unit tests for ServerSettings: the one reader of server environment variables."""

from __future__ import annotations

from pathlib import Path

import pytest

from miniagent.api.settings import ServerSettings
from miniagent.core.exceptions import ConfigurationError


class TestDefaults:
    def test_should_use_defaults_when_nothing_is_set(self) -> None:
        settings = ServerSettings.from_environment({})

        assert settings == ServerSettings()
        assert (settings.host, settings.port) == ("0.0.0.0", 8000)
        assert (settings.log_level, settings.reload) == ("info", False)
        assert settings.agents_config_path == Path("agents.yaml")
        assert settings.session_db_path is None
        assert settings.cors_origins == ()


class TestFromEnvironment:
    def test_should_read_every_variable(self) -> None:
        settings = ServerSettings.from_environment(
            {
                "HOST": "127.0.0.1",
                "PORT": "9001",
                "LOG_LEVEL": "debug",
                "RELOAD": "True",
                "AGENTS_CONFIG_PATH": "config/agents.yaml",
                "SESSION_DB_PATH": "data/sessions.db",
                "CORS_ORIGINS": " http://localhost:3000 , https://app.example.com ,",
            }
        )

        assert settings == ServerSettings(
            host="127.0.0.1",
            port=9001,
            log_level="debug",
            reload=True,
            agents_config_path=Path("config/agents.yaml"),
            session_db_path=Path("data/sessions.db"),
            cors_origins=("http://localhost:3000", "https://app.example.com"),
        )

    def test_should_read_the_process_environment_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PORT", "8123")
        assert ServerSettings.from_environment().port == 8123

    @pytest.mark.parametrize("value", ["", "false", "0", "yes"])
    def test_should_enable_reload_only_for_true(self, value: str) -> None:
        assert ServerSettings.from_environment({"RELOAD": value}).reload is False

    def test_should_treat_empty_paths_as_unset(self) -> None:
        settings = ServerSettings.from_environment(
            {"AGENTS_CONFIG_PATH": "", "SESSION_DB_PATH": ""}
        )
        assert settings.agents_config_path == Path("agents.yaml")
        assert settings.session_db_path is None

    def test_should_reject_a_port_that_is_not_a_number(self) -> None:
        with pytest.raises(ConfigurationError, match="PORT must be a whole number, got 'http'"):
            ServerSettings.from_environment({"PORT": "http"})
