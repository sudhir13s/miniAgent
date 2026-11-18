"""Shared naming rule for agent and template identifiers.

Agents and templates are both referenced by name from YAML, so they
follow one rule: letters, digits, hyphens, and underscores only.
"""

from __future__ import annotations


def validate_identifier(value: str, kind: str) -> str:
    """Ensure an identifier contains only letters, digits, hyphens, and underscores.

    Args:
        value: The identifier to validate.
        kind: What the identifier names (e.g. ``"Agent"``), for the error message.

    Returns:
        The validated identifier.

    Raises:
        ValueError: If the identifier contains any other character.
    """
    if not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            f"{kind} name '{value}' must contain only alphanumeric "
            f"characters, hyphens, and underscores."
        )
    return value
