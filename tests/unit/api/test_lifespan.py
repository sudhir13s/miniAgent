"""Tests for the API lifespan that opens configured MCP toolsets.

The end-to-end path with a real server lives in tests/integration/test_mcp_stdio.py;
these tests cover startup decisions without needing the ``mcp`` extra.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from miniagent.api.app import create_app
from miniagent.core.exceptions import ConfigurationError

_AGENTS_ONLY = "agents:\n  - name: bot\n    model: {id: openai/gpt-4o}\n"
_WITH_SERVER = "mcp_servers:\n  - name: echo\n    command: echo-server\n" + _AGENTS_ONLY


def _use_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str | None) -> None:
    config_file = tmp_path / "agents.yaml"
    if content is not None:
        config_file.write_text(content, encoding="utf-8")
    monkeypatch.setenv("AGENTS_CONFIG_PATH", str(config_file))


def _toolsets_during_lifespan(app: FastAPI) -> list[object]:
    with TestClient(app):
        return list(app.state.toolsets)


class TestLifespan:
    def test_missing_config_file_starts_without_toolsets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _use_config(monkeypatch, tmp_path, content=None)
        assert _toolsets_during_lifespan(create_app()) == []

    def test_config_without_mcp_servers_starts_without_toolsets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _use_config(monkeypatch, tmp_path, _AGENTS_ONLY)
        assert _toolsets_during_lifespan(create_app()) == []

    def test_servers_without_the_mcp_extra_fail_startup_with_install_hint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _use_config(monkeypatch, tmp_path, _WITH_SERVER)
        # A None entry makes the import raise ImportError, as if the extra were absent.
        monkeypatch.setitem(sys.modules, "miniagent.tool.mcp", None)

        with pytest.raises(ConfigurationError, match=r"miniagent\[mcp\]"):
            _toolsets_during_lifespan(create_app())

    def test_invalid_server_definition_fails_startup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _use_config(monkeypatch, tmp_path, "mcp_servers:\n  - name: broken\n" + _AGENTS_ONLY)

        with pytest.raises(ConfigurationError, match="MCP server at index 0"):
            _toolsets_during_lifespan(create_app())

    def test_toolsets_reset_after_shutdown(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _use_config(monkeypatch, tmp_path, _AGENTS_ONLY)
        app = create_app()
        _toolsets_during_lifespan(app)
        assert app.state.toolsets == []
