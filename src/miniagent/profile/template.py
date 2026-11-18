"""Pydantic model for reusable agent profile templates.

A ``ProfileTemplate`` captures what defines a *type* of agent — its
persona (system prompt), default tools, and iteration budget — without
binding it to a model. Passing a template to ``AgentProfile`` fills in
any field the caller does not set explicitly::

    from miniagent.profile.templates import WEB_RESEARCHER

    profile = AgentProfile(
        name="market-scout",
        template=WEB_RESEARCHER,
        model=ModelConfig(id="openai/gpt-4o-mini"),
    )

Built-in templates live in ``miniagent.profile.templates``; user-defined templates
are plain ``ProfileTemplate`` instances.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from miniagent.core.naming import validate_identifier


class ProfileTemplate(BaseModel):
    """Reusable, model-agnostic defaults for a type of agent.

    Attributes:
        name: Unique identifier, used for lookup from YAML.
        description: Human-readable summary of what this agent type does.
        system_prompt: System-level instructions defining the agent's persona.
        tools: Names of the tools this agent type is designed to use.
        max_iterations: Default reasoning-iteration budget.
    """

    name: str = Field(
        ...,
        min_length=1,
        description="Unique identifier, used for lookup from YAML.",
    )
    description: str = Field(
        default="",
        description="Human-readable summary of what this agent type does.",
    )
    system_prompt: str = Field(
        ...,
        min_length=1,
        description="System-level instructions defining the agent's persona.",
    )
    tools: tuple[str, ...] = Field(
        default=(),
        description="Names of the tools this agent type is designed to use.",
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default reasoning-iteration budget.",
    )

    model_config = {"frozen": True}

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, value: str) -> str:
        """Ensure the template name follows the shared identifier rule.

        Args:
            value: The template name to validate.

        Returns:
            The validated template name.

        Raises:
            ValueError: If the name contains invalid characters.
        """
        return validate_identifier(value, "Template")

    def profile_defaults(self) -> dict[str, Any]:
        """Return the ``AgentProfile`` field defaults this template provides.

        Returns:
            Mapping of ``AgentProfile`` field name to default value.
        """
        return {
            "description": self.description,
            "system_prompt": self.system_prompt,
            "tools": list(self.tools),
            "max_iterations": self.max_iterations,
        }
