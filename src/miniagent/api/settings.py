"""Server settings: every environment variable the API reads, defined in one place.

No other module reads the process environment for server configuration. Each
variable's name, default and parsing live here, so adding or renaming one is
a change to this file only::

    settings = ServerSettings.from_environment()
    settings.port          # DEFAULT_PORT unless PORT is set

(MCP server secrets, written as ``${VAR}`` in agents.yaml, are a different
thing: they are named by the user and resolved in ``profile.mcp_server_config``.)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from miniagent.core.exceptions import ConfigurationError

HOST_VARIABLE = "HOST"
PORT_VARIABLE = "PORT"
LOG_LEVEL_VARIABLE = "LOG_LEVEL"
RELOAD_VARIABLE = "RELOAD"
AGENTS_CONFIG_PATH_VARIABLE = "AGENTS_CONFIG_PATH"
SESSION_DB_PATH_VARIABLE = "SESSION_DB_PATH"
CORS_ORIGINS_VARIABLE = "CORS_ORIGINS"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_LOG_LEVEL = "info"
DEFAULT_AGENTS_CONFIG_PATH = Path("agents.yaml")

_TRUE = "true"
_LIST_SEPARATOR = ","


@dataclass(frozen=True)
class ServerSettings:
    """How the API server runs. Field defaults apply when a variable is unset.

    Attributes:
        host: Bind address (``HOST``).
        port: Bind port (``PORT``).
        log_level: Uvicorn log level (``LOG_LEVEL``).
        reload: Restart on code changes, for development (``RELOAD=true``).
        agents_config_path: The agents and MCP servers file (``AGENTS_CONFIG_PATH``).
        session_db_path: SQLite file for sessions; ``None`` keeps them in
            memory (``SESSION_DB_PATH``).
        cors_origins: Browser origins allowed to call the API; empty blocks
            cross-origin calls (``CORS_ORIGINS``, comma-separated).
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    log_level: str = DEFAULT_LOG_LEVEL
    reload: bool = False
    agents_config_path: Path = DEFAULT_AGENTS_CONFIG_PATH
    session_db_path: Path | None = None
    cors_origins: tuple[str, ...] = ()

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> ServerSettings:
        """Read the settings from environment variables.

        Args:
            environment: The variables to read; defaults to the process environment.

        Returns:
            The settings, with defaults for every unset variable.

        Raises:
            ConfigurationError: If ``PORT`` is not a whole number.
        """
        variables = os.environ if environment is None else environment
        defaults = cls()
        session_db_path = variables.get(SESSION_DB_PATH_VARIABLE)
        return cls(
            host=variables.get(HOST_VARIABLE, defaults.host),
            port=_parse_port(variables.get(PORT_VARIABLE), defaults.port),
            log_level=variables.get(LOG_LEVEL_VARIABLE, defaults.log_level),
            reload=variables.get(RELOAD_VARIABLE, "").strip().lower() == _TRUE,
            agents_config_path=Path(
                variables.get(AGENTS_CONFIG_PATH_VARIABLE) or defaults.agents_config_path
            ),
            session_db_path=Path(session_db_path) if session_db_path else None,
            cors_origins=_parse_list(variables.get(CORS_ORIGINS_VARIABLE, "")),
        )


def _parse_port(raw_port: str | None, default: int) -> int:
    """Return the port as a number, or ``default`` when unset."""
    if not raw_port:
        return default
    try:
        return int(raw_port)
    except ValueError as exc:
        raise ConfigurationError(
            f"{PORT_VARIABLE} must be a whole number, got {raw_port!r}.", cause=exc
        ) from exc


def _parse_list(raw_list: str) -> tuple[str, ...]:
    """Split a comma-separated value, dropping blanks and surrounding spaces."""
    return tuple(item.strip() for item in raw_list.split(_LIST_SEPARATOR) if item.strip())
