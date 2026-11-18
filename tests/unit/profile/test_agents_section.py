"""Tests for reading the ``agents:`` section of a configuration mapping."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from miniagent.core.exceptions import ConfigurationError
from miniagent.profile.agents_section import (
    agent_entries,
    resolve_template,
    select_by_name,
    template_lookup,
)
from miniagent.profile.template import ProfileTemplate
from miniagent.profile.templates import WEB_RESEARCHER


@dataclass(frozen=True)
class _Named:
    name: str


class TestAgentEntries:
    def test_should_return_the_entries(self) -> None:
        assert agent_entries({"agents": [{"name": "a"}]}, "test") == [{"name": "a"}]

    def test_should_reject_a_missing_section(self) -> None:
        with pytest.raises(
            ConfigurationError, match="Missing 'agents' key in configuration from test"
        ):
            agent_entries({"name": "a"}, "test")

    def test_should_reject_an_empty_list(self) -> None:
        with pytest.raises(ConfigurationError, match="empty"):
            agent_entries({"agents": []}, "test")

    def test_should_reject_a_non_list(self) -> None:
        with pytest.raises(ConfigurationError, match="must be a list"):
            agent_entries({"agents": {"name": "a"}}, "test")

    def test_should_reject_entries_that_are_not_mappings(self) -> None:
        with pytest.raises(ConfigurationError, match="Agent at index 1 must be a mapping"):
            agent_entries({"agents": [{"name": "a"}, "just a string"]}, "test")


class TestTemplates:
    def test_lookup_should_let_custom_templates_override_builtins(self) -> None:
        custom = ProfileTemplate(name="web-researcher", system_prompt="Mine.")
        assert template_lookup([custom])["web-researcher"] is custom
        assert "math-solver" in template_lookup([])

    def test_should_replace_a_template_name_with_the_template(self) -> None:
        resolved = resolve_template({"template": "web-researcher"}, template_lookup([]), "here")
        assert resolved["template"] is WEB_RESEARCHER

    def test_should_leave_entries_without_a_template_alone(self) -> None:
        entry = {"name": "a"}
        assert resolve_template(entry, template_lookup([]), "here") is entry

    def test_should_list_available_templates_for_an_unknown_name(self) -> None:
        with pytest.raises(
            ConfigurationError, match=r"Unknown template 'nope' for here.*math-solver"
        ):
            resolve_template({"template": "nope"}, template_lookup([]), "here")


class TestSelectByName:
    def test_should_return_the_named_item(self) -> None:
        items = [_Named("a"), _Named("b")]
        assert select_by_name(items, "b", "test") is items[1]

    def test_should_list_defined_names_when_missing(self) -> None:
        with pytest.raises(
            ConfigurationError, match=r"Agent 'c' is not defined.*Defined agents: a, b"
        ):
            select_by_name([_Named("a"), _Named("b")], "c", "test")
