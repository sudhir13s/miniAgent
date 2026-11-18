"""Declarative MCP server definitions: the ``mcp_servers:`` section of agents.yaml.

Each entry describes how to reach one MCP server; agents select servers by
name with ``toolsets: [name]``. Load them with
``MCPServerConfig.from_yaml("agents.yaml")``::

    mcp_servers:
      - name: github
        command: github-mcp-server
        args: [stdio]
        env: {GITHUB_TOKEN: "${GITHUB_TOKEN}"}
      - name: docs
        url: https://docs.example.com/mcp
        headers: {Authorization: "Bearer ${DOCS_TOKEN}"}

Secrets stay out of the file: ``${VAR}`` in ``env`` and ``headers`` values is
replaced from the process environment only when the toolset is built
(``MCPToolset.from_config``), so the loaded configuration never holds the
secret itself.

This module has no dependency on the ``mcp`` package; connecting a server
needs the ``mcp`` extra (see ``miniagent.tool.mcp``).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from miniagent.core.exceptions import ConfigurationError
from miniagent.core.naming import validate_identifier
from miniagent.profile.config_file import DICT_SOURCE, read_yaml_mapping
from miniagent.tool.base import DEFAULT_TOOL_TIMEOUT_SECONDS

_ENVIRONMENT_REFERENCE = re.compile(r"\$\{(\w+)\}")


class MCPServerConfig(BaseModel):
    """How to reach one MCP server, and which of its tools to expose.

    Exactly one of ``command`` (a stdio subprocess) or ``url`` (Streamable
    HTTP) must be set.

    Attributes:
        name: Identifier agents use in ``toolsets: [name]``.
        command: Executable that starts a stdio server.
        args: Arguments for ``command``.
        env: Environment for the subprocess; values may contain ``${VAR}``.
        cwd: Working directory for the subprocess.
        url: Streamable HTTP endpoint of a remote server.
        headers: Request headers for ``url``; values may contain ``${VAR}``.
        tool_filter: Server tool names to expose; ``None`` exposes all.
        name_prefix: Prepended to every tool name.
        timeout_seconds: Per-call limit for every tool; ``None`` disables it.
    """

    name: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    tool_filter: list[str] | None = None
    name_prefix: str = ""
    timeout_seconds: float | None = Field(default=DEFAULT_TOOL_TIMEOUT_SECONDS, gt=0)

    model_config = {"frozen": True, "extra": "forbid"}

    @property
    def transport(self) -> Literal["stdio", "http"]:
        """``"stdio"`` for a subprocess server, ``"http"`` for a remote one."""
        return "stdio" if self.command is not None else "http"

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, value: str) -> str:
        """Ensure the server name follows the shared identifier rule."""
        return validate_identifier(value, "MCP server")

    @model_validator(mode="after")
    def validate_transport_fields(self) -> MCPServerConfig:
        """Ensure exactly one transport is chosen and no field belongs to the other."""
        if (self.command is None) == (self.url is None):
            raise ValueError("Set either 'command' (stdio server) or 'url' (HTTP server).")
        if self.command is not None and self.headers:
            raise ValueError("'headers' only applies to HTTP servers (set with 'url').")
        if self.url is not None and (self.args or self.env or self.cwd):
            raise ValueError("'args', 'env' and 'cwd' only apply to stdio servers.")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> list[MCPServerConfig]:
        """Load the servers in a YAML file's optional ``mcp_servers:`` section.

        Args:
            path: The ``.yaml`` or ``.yml`` configuration file.

        Returns:
            The server definitions; empty when the file defines none.

        Raises:
            ConfigurationError: If the file is unreadable or invalid, or a
                server definition is invalid.
        """
        return parse_mcp_servers(read_yaml_mapping(path), str(path))

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> list[MCPServerConfig]:
        """Load the servers from a configuration mapping shaped like the YAML file.

        Args:
            config: Mapping with an optional ``mcp_servers`` list.

        Returns:
            The server definitions; empty when ``config`` defines none.

        Raises:
            ConfigurationError: If a server definition is invalid.
        """
        return parse_mcp_servers(config, DICT_SOURCE)

    def resolved_env(self) -> dict[str, str]:
        """Return ``env`` with every ``${VAR}`` replaced from the process environment.

        Raises:
            ConfigurationError: If a referenced variable is not set.
        """
        return self._expand_values(self.env)

    def resolved_headers(self) -> dict[str, str]:
        """Return ``headers`` with every ``${VAR}`` replaced from the process environment.

        Raises:
            ConfigurationError: If a referenced variable is not set.
        """
        return self._expand_values(self.headers)

    def _expand_values(self, values: Mapping[str, str]) -> dict[str, str]:
        return {key: self._expand(value) for key, value in values.items()}

    def _expand(self, value: str) -> str:
        def substitute(match: re.Match[str]) -> str:
            variable = match.group(1)
            if variable not in os.environ:
                raise ConfigurationError(
                    f"Environment variable {variable} is not set; "
                    f"MCP server '{self.name}' needs it."
                )
            return os.environ[variable]

        return _ENVIRONMENT_REFERENCE.sub(substitute, value)


def parse_mcp_servers(raw_config: Mapping[str, Any], source: str) -> list[MCPServerConfig]:
    """Validate the optional ``mcp_servers`` list of a parsed configuration.

    Args:
        raw_config: The parsed YAML mapping.
        source: Description of the configuration source, for error messages.

    Returns:
        The server definitions; empty when the section is absent.

    Raises:
        ConfigurationError: If the section is not a list, an entry is
            invalid, or two servers share a name.
    """
    entries = raw_config.get("mcp_servers", [])
    if not isinstance(entries, list):
        raise ConfigurationError(
            f"'mcp_servers' must be a list in configuration from {source}, "
            f"got: {type(entries).__name__}"
        )

    servers: list[MCPServerConfig] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigurationError(
                f"MCP server at index {index} must be a mapping (source: {source})."
            )
        try:
            server = MCPServerConfig(**entry)
        except ValidationError as exc:
            raise ConfigurationError(
                f"Validation failed for MCP server at index {index} (source: {source}): {exc}",
                cause=exc,
            ) from exc
        if any(existing.name == server.name for existing in servers):
            raise ConfigurationError(
                f"Duplicate MCP server name '{server.name}' in configuration from {source}."
            )
        servers.append(server)
    return servers
