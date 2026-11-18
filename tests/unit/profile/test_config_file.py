"""Tests for reading the YAML configuration file."""

from __future__ import annotations

from pathlib import Path

import pytest

from miniagent.core.exceptions import ConfigurationError
from miniagent.profile.config_file import read_yaml_mapping


def _write(tmp_path: Path, filename: str, content: str) -> Path:
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    return path


class TestReadYamlMapping:
    @pytest.mark.parametrize("filename", ["agents.yaml", "agents.yml"])
    def test_should_read_a_top_level_mapping(self, tmp_path: Path, filename: str) -> None:
        path = _write(tmp_path, filename, "agents:\n  - name: a\n")
        assert read_yaml_mapping(path) == {"agents": [{"name": "a"}]}

    def test_should_accept_a_string_path(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "agents.yaml", "key: value\n")
        assert read_yaml_mapping(str(path)) == {"key": "value"}

    def test_should_reject_a_missing_file(self) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            read_yaml_mapping("/nonexistent/agents.yaml")

    def test_should_reject_other_extensions(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "agents.json", "{}")
        with pytest.raises(ConfigurationError, match=r"must be \.yaml or \.yml"):
            read_yaml_mapping(path)

    def test_should_reject_malformed_yaml(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "bad.yaml", "[ invalid: yaml: {")
        with pytest.raises(ConfigurationError, match="Failed to parse"):
            read_yaml_mapping(path)

    def test_should_reject_a_non_mapping_top_level(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "list.yaml", "- item1\n- item2\n")
        with pytest.raises(ConfigurationError, match="Expected a YAML mapping"):
            read_yaml_mapping(path)
