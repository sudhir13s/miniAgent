"""Read miniAgent's YAML configuration file.

One file holds every section (``agents:``, ``mcp_servers:``); each
configuration class reads the section it owns through ``read_yaml_mapping``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from miniagent.core.exceptions import ConfigurationError

DICT_SOURCE = "<dict>"
"""How error messages describe configuration that came from a dict, not a file."""

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})


def read_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Read a YAML file whose top level must be a mapping.

    Args:
        path: The ``.yaml`` or ``.yml`` file.

    Returns:
        The parsed top-level mapping.

    Raises:
        ConfigurationError: If the file is missing, not ``.yaml``/``.yml``,
            unreadable, invalid YAML, or not a mapping at the top level.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")
    if file_path.suffix not in _YAML_SUFFIXES:
        raise ConfigurationError(
            f"Configuration file must be .yaml or .yml, got: {file_path.suffix}"
        )

    try:
        raw_config = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(
            f"Failed to read configuration file: {file_path}", cause=exc
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse YAML in: {file_path}", cause=exc) from exc

    if not isinstance(raw_config, dict):
        raise ConfigurationError(
            f"Expected a YAML mapping at the top level, got: {type(raw_config).__name__}"
        )
    return raw_config
