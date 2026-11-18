"""Read the ``agents:`` section of a configuration mapping.

Structural checks and template-name resolution for ``AgentProfile.from_yaml``
and ``from_dict``; validating each entry is left to ``AgentProfile`` itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, TypeVar

from miniagent.core.exceptions import ConfigurationError
from miniagent.profile.template import ProfileTemplate
from miniagent.profile.templates.catalog import BUILTIN_TEMPLATES


class _Named(Protocol):
    @property
    def name(self) -> str: ...


NamedT = TypeVar("NamedT", bound=_Named)


def agent_entries(config: Mapping[str, Any], source: str) -> list[dict[str, Any]]:
    """Return the ``agents`` entries, which must be a non-empty list of mappings.

    Raises:
        ConfigurationError: If the section is missing, not a list, empty, or
            holds an entry that is not a mapping.
    """
    entries = config.get("agents")
    if entries is None:
        raise ConfigurationError(
            f"Missing 'agents' key in configuration from {source}. "
            "For a single agent's fields, use AgentProfile(...) instead."
        )
    if not isinstance(entries, list):
        raise ConfigurationError(
            f"'agents' must be a list in configuration from {source}, got: {type(entries).__name__}"
        )
    if not entries:
        raise ConfigurationError(f"'agents' list is empty in configuration from {source}.")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigurationError(
                f"Agent at index {index} must be a mapping, "
                f"got: {type(entry).__name__} (source: {source})"
            )
    return entries


def template_lookup(templates: Iterable[ProfileTemplate]) -> dict[str, ProfileTemplate]:
    """Merge built-in and custom templates by name; custom ones win."""
    return {**BUILTIN_TEMPLATES, **{template.name: template for template in templates}}


def resolve_template(
    entry: dict[str, Any], lookup: Mapping[str, ProfileTemplate], location: str
) -> dict[str, Any]:
    """Replace a template name in an agent entry with the template object.

    Raises:
        ConfigurationError: If the named template does not exist.
    """
    template_name = entry.get("template")
    if not isinstance(template_name, str):
        return entry
    template = lookup.get(template_name)
    if template is None:
        available = ", ".join(sorted(lookup)) or "none"
        raise ConfigurationError(
            f"Unknown template '{template_name}' for {location}. Available templates: {available}."
        )
    return {**entry, "template": template}


def select_by_name(items: Sequence[NamedT], name: str, source: str) -> NamedT:
    """Return the agent called ``name``.

    Raises:
        ConfigurationError: If no agent has that name; the defined names are listed.
    """
    for item in items:
        if item.name == name:
            return item
    available = ", ".join(item.name for item in items)
    raise ConfigurationError(
        f"Agent '{name}' is not defined in configuration from {source}. "
        f"Defined agents: {available}."
    )
