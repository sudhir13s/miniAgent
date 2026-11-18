"""Tests for ``MCPServerConfig`` and the ``mcp_servers:`` YAML section."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from miniagent.core.exceptions import ConfigurationError
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.mcp_server_config import MCPServerConfig

_STDIO: dict[str, Any] = {"name": "github", "command": "github-mcp-server", "args": ["stdio"]}
_HTTP: dict[str, Any] = {"name": "docs", "url": "https://docs.example.com/mcp"}


class TestMCPServerConfig:
    def test_stdio_server(self) -> None:
        config = MCPServerConfig(**_STDIO)
        assert config.transport == "stdio"
        assert config.args == ["stdio"]

    def test_http_server(self) -> None:
        config = MCPServerConfig(**_HTTP, headers={"X-Team": "core"})
        assert config.transport == "http"
        assert config.headers == {"X-Team": "core"}

    def test_tool_options_have_defaults(self) -> None:
        config = MCPServerConfig(**_STDIO)
        assert config.tool_filter is None
        assert config.name_prefix == ""
        assert config.timeout_seconds == 60

    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({"name": "x"}, "either 'command'"),
            ({"name": "x", "command": "a", "url": "https://b"}, "either 'command'"),
            ({**_STDIO, "headers": {"A": "1"}}, "'headers' only applies"),
            ({**_HTTP, "args": ["--flag"]}, "'args', 'env' and 'cwd' only apply"),
            ({**_HTTP, "env": {"A": "1"}}, "'args', 'env' and 'cwd' only apply"),
            ({**_STDIO, "name": "bad name"}, "MCP server name"),
            ({**_STDIO, "timeout_seconds": 0}, "greater than 0"),
            ({**_STDIO, "unknown": True}, "Extra inputs"),
        ],
    )
    def test_invalid_definitions_are_rejected(self, fields: dict[str, Any], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            MCPServerConfig(**fields)


class TestEnvironmentExpansion:
    def test_env_and_header_values_expand_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIAGENT_TEST_TOKEN", "abc123")
        stdio = MCPServerConfig(**_STDIO, env={"TOKEN": "${MINIAGENT_TEST_TOKEN}"})
        http = MCPServerConfig(**_HTTP, headers={"Authorization": "Bearer ${MINIAGENT_TEST_TOKEN}"})

        assert stdio.resolved_env() == {"TOKEN": "abc123"}
        assert http.resolved_headers() == {"Authorization": "Bearer abc123"}

    def test_values_without_variables_are_unchanged(self) -> None:
        assert MCPServerConfig(**_STDIO, env={"MODE": "read-only"}).resolved_env() == {
            "MODE": "read-only"
        }

    def test_stored_config_keeps_the_placeholder_not_the_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MINIAGENT_TEST_TOKEN", "abc123")
        config = MCPServerConfig(**_STDIO, env={"TOKEN": "${MINIAGENT_TEST_TOKEN}"})
        assert "abc123" not in config.model_dump_json()

    def test_missing_variable_names_it_without_a_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MINIAGENT_TEST_MISSING", raising=False)
        config = MCPServerConfig(**_STDIO, env={"TOKEN": "${MINIAGENT_TEST_MISSING}"})

        with pytest.raises(ConfigurationError, match=r"MINIAGENT_TEST_MISSING.*github"):
            config.resolved_env()


class TestLoadMCPServers:
    def test_missing_section_means_no_servers(self) -> None:
        assert MCPServerConfig.from_dict({"agents": []}) == []

    def test_parses_each_entry(self) -> None:
        servers = MCPServerConfig.from_dict({"mcp_servers": [_STDIO, _HTTP]})
        assert [server.name for server in servers] == ["github", "docs"]

    def test_section_must_be_a_list(self) -> None:
        with pytest.raises(ConfigurationError, match="'mcp_servers' must be a list"):
            MCPServerConfig.from_dict({"mcp_servers": _STDIO})

    def test_invalid_entry_reports_its_index(self) -> None:
        with pytest.raises(ConfigurationError, match="MCP server at index 1"):
            MCPServerConfig.from_dict({"mcp_servers": [_STDIO, {"name": "x"}]})

    def test_duplicate_names_are_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="Duplicate MCP server name 'github'"):
            MCPServerConfig.from_dict({"mcp_servers": [_STDIO, _STDIO]})

    def test_loads_from_yaml_file_next_to_agents(self, tmp_path: Path) -> None:
        config_file = tmp_path / "agents.yaml"
        config_file.write_text(
            "mcp_servers:\n"
            "  - name: github\n"
            "    command: github-mcp-server\n"
            "    args: [stdio]\n"
            "    tool_filter: [create_issue]\n"
            "agents:\n"
            "  - name: triager\n"
            "    model: {id: openai/gpt-4o}\n"
            "    toolsets: [github]\n",
            encoding="utf-8",
        )

        [server] = MCPServerConfig.from_yaml(config_file)
        [profile] = AgentProfile.from_yaml(config_file)

        assert server.tool_filter == ["create_issue"]
        assert profile.toolsets == ["github"]
