"""Tests for profile templates: the ProfileTemplate model, template merging
in AgentProfile, the built-in catalog, and template resolution when
loading with AgentProfile.from_yaml / from_dict.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from miniagent.core.exceptions import ConfigurationError
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.model_config import ModelConfig
from miniagent.profile.template import ProfileTemplate
from miniagent.profile.templates import (
    BUILTIN_TEMPLATES,
    TOOL_ASSISTANT,
    WEB_RESEARCHER,
    WEB_RESEARCHER_PROMPT,
)
from miniagent.tool import (
    CalculatorTool,
    EchoTool,
    FileReaderTool,
    JsonFormatterTool,
    TextTransformTool,
    TimestampTool,
    UuidTool,
    WeatherTool,
    WebFetchTool,
    WordCountTool,
)

_MODEL = ModelConfig(id="openai/gpt-4o-mini")


def _builtin_tool_names() -> set[str]:
    tool_classes = (
        CalculatorTool,
        EchoTool,
        FileReaderTool,
        JsonFormatterTool,
        TextTransformTool,
        TimestampTool,
        UuidTool,
        WeatherTool,
        WebFetchTool,
        WordCountTool,
    )
    return {tool_class().schema.name for tool_class in tool_classes}


def _config_with_template(template_name: str, **fields: object) -> dict[str, object]:
    agent = {
        "name": "scout",
        "template": template_name,
        "model": {"id": "openai/gpt-4o"},
        **fields,
    }
    return {"agents": [agent]}


def _custom_template(name: str = "support-agent") -> ProfileTemplate:
    return ProfileTemplate(
        name=name,
        description="Answers customer questions.",
        system_prompt="You are a friendly customer-support agent.",
        tools=("web_fetch",),
        max_iterations=4,
    )


# =====================================================================
# ProfileTemplate model
# =====================================================================


class TestProfileTemplate:
    def test_should_create_template_when_required_fields_given(self) -> None:
        template = ProfileTemplate(name="helper", system_prompt="Be helpful.")
        assert template.description == ""
        assert template.tools == ()
        assert template.max_iterations == 10

    @pytest.mark.parametrize("name", ["web-researcher", "web_researcher", "Researcher2"])
    def test_should_accept_name_when_it_follows_agent_naming_rule(self, name: str) -> None:
        assert ProfileTemplate(name=name, system_prompt="Be helpful.").name == name

    @pytest.mark.parametrize("name", ["has space", "dots.name", "bang!"])
    def test_should_reject_name_when_it_breaks_agent_naming_rule(self, name: str) -> None:
        with pytest.raises(ValidationError, match="Template name"):
            ProfileTemplate(name=name, system_prompt="Be helpful.")
        with pytest.raises(ValidationError, match="Agent name"):
            AgentProfile(name=name, model=_MODEL)

    def test_should_reject_template_when_system_prompt_empty(self) -> None:
        with pytest.raises(ValidationError):
            ProfileTemplate(name="helper", system_prompt="")

    def test_should_be_immutable(self) -> None:
        template = _custom_template()
        with pytest.raises(ValidationError):
            template.system_prompt = "changed"  # type: ignore[misc]

    def test_should_return_profile_defaults_with_tools_as_list(self) -> None:
        assert _custom_template().profile_defaults() == {
            "description": "Answers customer questions.",
            "system_prompt": "You are a friendly customer-support agent.",
            "tools": ["web_fetch"],
            "max_iterations": 4,
        }


# =====================================================================
# AgentProfile + template merging
# =====================================================================


class TestAgentProfileWithTemplate:
    def test_should_fill_fields_from_template_when_not_given(self) -> None:
        profile = AgentProfile(name="scout", model=_MODEL, template=WEB_RESEARCHER)
        assert profile.system_prompt == WEB_RESEARCHER_PROMPT
        assert profile.description == WEB_RESEARCHER.description
        assert profile.tools == ["web_fetch", "file_reader"]
        assert profile.max_iterations == WEB_RESEARCHER.max_iterations
        assert profile.template is WEB_RESEARCHER

    def test_should_prefer_explicit_fields_over_template(self) -> None:
        profile = AgentProfile(
            name="scout",
            model=_MODEL,
            template=WEB_RESEARCHER,
            system_prompt="Custom prompt.",
            max_iterations=20,
        )
        assert profile.system_prompt == "Custom prompt."
        assert profile.max_iterations == 20
        assert profile.tools == ["web_fetch", "file_reader"]

    def test_should_replace_template_tools_when_tools_given(self) -> None:
        profile = AgentProfile(
            name="scout", model=_MODEL, template=WEB_RESEARCHER, tools=["weather"]
        )
        assert profile.tools == ["weather"]

    def test_should_keep_empty_tools_when_explicitly_empty(self) -> None:
        profile = AgentProfile(name="scout", model=_MODEL, template=WEB_RESEARCHER, tools=[])
        assert profile.tools == []

    def test_should_accept_user_defined_template(self) -> None:
        profile = AgentProfile(name="support-bot", model=_MODEL, template=_custom_template())
        assert profile.system_prompt == "You are a friendly customer-support agent."
        assert profile.max_iterations == 4

    def test_should_accept_template_as_dict(self) -> None:
        profile = AgentProfile(
            name="helper",
            model=_MODEL,
            template={"name": "inline-helper", "system_prompt": "Inline prompt."},
        )
        assert profile.system_prompt == "Inline prompt."
        assert profile.template is not None
        assert profile.template.name == "inline-helper"

    def test_should_reject_template_given_by_name(self) -> None:
        with pytest.raises(ValidationError, match="from_yaml"):
            AgentProfile(name="scout", model=_MODEL, template="web-researcher")

    def test_should_reject_template_of_wrong_type(self) -> None:
        with pytest.raises(ValidationError):
            AgentProfile(name="scout", model=_MODEL, template=42)

    def test_should_use_profile_defaults_when_no_template(self) -> None:
        profile = AgentProfile(name="plain", model=_MODEL)
        assert profile.template is None
        assert profile.system_prompt == "You are a helpful AI assistant."
        assert profile.tools == []

    def test_should_define_every_template_field_on_agent_profile(self) -> None:
        """Symmetry guard: a template may only carry fields a profile has."""
        template_fields = set(ProfileTemplate.model_fields)
        profile_fields = set(AgentProfile.model_fields)
        assert template_fields - profile_fields == set()

    def test_should_provide_a_default_for_every_persona_field(self) -> None:
        persona_fields = set(ProfileTemplate.model_fields) - {"name"}
        assert set(_custom_template().profile_defaults()) == persona_fields

    def test_should_exclude_template_from_dump(self) -> None:
        profile = AgentProfile(name="scout", model=_MODEL, template=WEB_RESEARCHER)
        dumped = profile.model_dump()
        assert "template" not in dumped
        assert AgentProfile(**dumped).system_prompt == profile.system_prompt


# =====================================================================
# Built-in catalog
# =====================================================================


class TestBuiltinTemplates:
    def test_should_key_each_template_by_its_name(self) -> None:
        for name, template in BUILTIN_TEMPLATES.items():
            assert template.name == name

    def test_should_only_reference_existing_builtin_tools(self) -> None:
        available = _builtin_tool_names()
        for template in BUILTIN_TEMPLATES.values():
            unknown = set(template.tools) - available
            assert not unknown, f"{template.name} references unknown tools: {unknown}"

    def test_should_be_read_only(self) -> None:
        with pytest.raises(TypeError):
            BUILTIN_TEMPLATES["new"] = TOOL_ASSISTANT  # type: ignore[index]

    def test_should_build_valid_profile_from_every_template(self) -> None:
        for template in BUILTIN_TEMPLATES.values():
            profile = AgentProfile(name="agent", model=_MODEL, template=template)
            assert profile.system_prompt == template.system_prompt


# =====================================================================
# Template resolution when loading configuration
# =====================================================================


class TestTemplatesInConfig:
    def test_should_resolve_builtin_template_by_name(self) -> None:
        profiles = AgentProfile.from_dict(_config_with_template("web-researcher"))
        assert profiles[0].template is WEB_RESEARCHER
        assert profiles[0].tools == ["web_fetch", "file_reader"]

    def test_should_let_yaml_fields_override_template(self) -> None:
        profiles = AgentProfile.from_dict(_config_with_template("web-researcher", max_iterations=3))
        assert profiles[0].max_iterations == 3

    def test_should_resolve_custom_template_by_name(self) -> None:
        profiles = AgentProfile.from_dict(
            _config_with_template("support-agent"), templates=[_custom_template()]
        )
        assert profiles[0].system_prompt == "You are a friendly customer-support agent."

    def test_should_prefer_custom_template_over_builtin_with_same_name(self) -> None:
        override = _custom_template(name="web-researcher")
        profiles = AgentProfile.from_dict(
            _config_with_template("web-researcher"), templates=[override]
        )
        assert profiles[0].template is override

    def test_should_raise_when_template_unknown(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown template 'nope'"):
            AgentProfile.from_dict(_config_with_template("nope"))

    def test_should_resolve_template_when_loading_from_file(self, tmp_path: Path) -> None:
        config = tmp_path / "agents.yaml"
        config.write_text(
            "agents:\n"
            "  - name: summarizer\n"
            "    template: document-summarizer\n"
            "    model:\n"
            "      id: openai/gpt-4o-mini\n",
            encoding="utf-8",
        )
        profiles = AgentProfile.from_yaml(config)
        assert profiles[0].tools == ["file_reader", "word_count"]
