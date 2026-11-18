"""Declarative agent profile: identity, persona, model, and tools.

An ``AgentProfile`` is built in code, optionally from a ``ProfileTemplate``
that supplies the persona fields, or loaded from configuration::

    profiles = AgentProfile.from_yaml("agents.yaml")                 # all agents
    researcher = AgentProfile.from_yaml("agents.yaml", name="researcher")
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, overload

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from miniagent.core.exceptions import ConfigurationError
from miniagent.core.naming import validate_identifier
from miniagent.profile.agents_section import (
    agent_entries,
    resolve_template,
    select_by_name,
    template_lookup,
)
from miniagent.profile.config_file import DICT_SOURCE, read_yaml_mapping
from miniagent.profile.model_config import ModelConfig
from miniagent.profile.template import ProfileTemplate


class AgentProfile(BaseModel):
    """Declarative profile defining an agent's identity and capabilities.

    Loaded from YAML configuration files, each profile specifies the
    agent's persona, its designated LLM, and the tools it may use.

    Only ``name`` and ``model`` are required; all other fields have
    sensible defaults so minimal YAML configs work out of the box.

    A ``template`` supplies defaults for ``description``, ``system_prompt``,
    ``tools``, and ``max_iterations``. Precedence, highest first: fields
    passed explicitly, then the template, then this model's defaults.
    An explicit ``tools`` list replaces the template's tools entirely.

    Attributes:
        name: Unique identifier for this agent.
        model: The LLM configuration for this specific agent.
        template: Optional template the profile was built from.
        description: Human-readable summary of the agent's purpose.
        system_prompt: System-level instructions provided to the LLM.
        tools: List of tool names this agent is permitted to use.
        toolsets: Names of toolsets (e.g. MCP servers) whose tools the agent
            discovers at run time.
        max_iterations: Maximum reasoning iterations before stopping.
        max_context_tokens: Token budget for the messages sent on each model
            call; ``None`` sends the full history.
    """

    name: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this agent.",
    )
    model: ModelConfig = Field(
        ...,
        description="The LLM configuration for this specific agent.",
    )
    template: ProfileTemplate | None = Field(
        default=None,
        exclude=True,
        description="Optional template supplying defaults for unset fields.",
    )
    description: str = Field(
        default="",
        description="Human-readable summary of the agent's purpose.",
    )
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="System-level instructions provided to the LLM.",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="List of tool names this agent is permitted to use.",
    )
    toolsets: list[str] = Field(
        default_factory=list,
        description="Names of toolsets (e.g. MCP servers) whose tools are discovered at run time.",
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum reasoning iterations before stopping.",
    )
    max_context_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Token budget for the messages sent on each model call; older turns are "
            "dropped to fit. None sends the full history."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def apply_template(cls, data: Any) -> Any:
        """Fill fields the caller did not set from the given template.

        Args:
            data: Raw constructor input.

        Returns:
            The input with template defaults merged beneath explicit fields.

        Raises:
            ValueError: If ``template`` is given as a name instead of an object.
        """
        if not isinstance(data, dict) or data.get("template") is None:
            return data

        template = data["template"]
        if isinstance(template, str):
            raise ValueError(
                f"Template '{template}' was given by name. Pass a ProfileTemplate "
                f"object (e.g. from miniagent.profile.templates); names are resolved only "
                f"by AgentProfile.from_yaml / from_dict."
            )
        if isinstance(template, dict):
            template = ProfileTemplate.model_validate(template)
        if not isinstance(template, ProfileTemplate):
            # Let field validation report the type error.
            return data

        return {**template.profile_defaults(), **data, "template": template}

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, value: str) -> str:
        """Ensure agent name contains only valid characters.

        Args:
            value: The agent name to validate.

        Returns:
            The validated agent name.

        Raises:
            ValueError: If the name contains invalid characters.
        """
        return validate_identifier(value, "Agent")

    @field_validator("toolsets")
    @classmethod
    def validate_toolset_names(cls, value: list[str]) -> list[str]:
        """Ensure every toolset name follows the shared identifier rule.

        Args:
            value: The toolset names to validate.

        Returns:
            The validated names.

        Raises:
            ValueError: If a name contains invalid characters.
        """
        return [validate_identifier(name, "Toolset") for name in value]

    model_config = {"frozen": True, "extra": "forbid"}

    @overload
    @classmethod
    def from_yaml(
        cls, path: str | Path, *, name: None = None, templates: Iterable[ProfileTemplate] = ()
    ) -> list[AgentProfile]: ...

    @overload
    @classmethod
    def from_yaml(
        cls, path: str | Path, *, name: str, templates: Iterable[ProfileTemplate] = ()
    ) -> AgentProfile: ...

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
        templates: Iterable[ProfileTemplate] = (),
    ) -> list[AgentProfile] | AgentProfile:
        """Load the agents defined in a YAML file's ``agents:`` section.

        Args:
            path: The ``.yaml`` or ``.yml`` configuration file.
            name: Return only the agent with this name instead of all of them.
            templates: Custom templates agents may name in ``template:``;
                they override built-in templates with the same name.

        Returns:
            Every profile in file order, or the one named ``name``.

        Raises:
            ConfigurationError: If the file is unreadable or invalid, an agent
                fails validation, names repeat, or ``name`` is not defined.
        """
        profiles = _parse_agents(read_yaml_mapping(path), str(path), templates)
        return profiles if name is None else select_by_name(profiles, name, str(path))

    @overload
    @classmethod
    def from_dict(
        cls,
        config: Mapping[str, Any],
        *,
        name: None = None,
        templates: Iterable[ProfileTemplate] = (),
    ) -> list[AgentProfile]: ...

    @overload
    @classmethod
    def from_dict(
        cls, config: Mapping[str, Any], *, name: str, templates: Iterable[ProfileTemplate] = ()
    ) -> AgentProfile: ...

    @classmethod
    def from_dict(
        cls,
        config: Mapping[str, Any],
        *,
        name: str | None = None,
        templates: Iterable[ProfileTemplate] = (),
    ) -> list[AgentProfile] | AgentProfile:
        """Load agents from a configuration mapping shaped like the YAML file.

        ``config`` is the whole configuration (``{"agents": [...]}``). For a
        single agent's fields, construct ``AgentProfile(...)`` directly.

        Args:
            config: Mapping with an ``agents`` list.
            name: Return only the agent with this name instead of all of them.
            templates: Custom templates agents may name in ``template:``.

        Returns:
            Every profile in order, or the one named ``name``.

        Raises:
            ConfigurationError: If the mapping is invalid, an agent fails
                validation, names repeat, or ``name`` is not defined.
        """
        profiles = _parse_agents(config, DICT_SOURCE, templates)
        return profiles if name is None else select_by_name(profiles, name, DICT_SOURCE)


def _parse_agents(
    config: Mapping[str, Any], source: str, templates: Iterable[ProfileTemplate]
) -> list[AgentProfile]:
    """Validate every entry of the ``agents`` section, rejecting repeated names."""
    lookup = template_lookup(templates)
    profiles: list[AgentProfile] = []
    for index, entry in enumerate(agent_entries(config, source)):
        location = f"agent at index {index} (source: {source})"
        try:
            profile = AgentProfile(**resolve_template(entry, lookup, location))
        except ValidationError as exc:
            raise ConfigurationError(f"Validation failed for {location}: {exc}", cause=exc) from exc
        if any(existing.name == profile.name for existing in profiles):
            raise ConfigurationError(
                f"Duplicate agent name '{profile.name}' found in configuration from {source}."
            )
        profiles.append(profile)
    return profiles
