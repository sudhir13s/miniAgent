"""Describe Pydantic validation errors in words a model can act on."""

from __future__ import annotations

from pydantic import ValidationError


def describe_validation_error(exc: ValidationError, root_label: str) -> str:
    """Summarise validation errors as ``field: problem`` pairs.

    Args:
        exc: The validation failure.
        root_label: Name used when the whole value is wrong rather than one
            field, e.g. ``"arguments"`` or ``"reply"``.

    Returns:
        One ``location: message`` pair per error, separated by ``"; "``.
    """
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or root_label}: {error['msg']}"
        for error in exc.errors()
    )
