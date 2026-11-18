"""FastAPI dependency injection for the miniAgent framework.

Provides dependency functions that resolve agent configurations,
agent instances, and workflow instances, enabling clean separation
between HTTP handling and business logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from miniagent.agent.base import BaseAgent
from miniagent.agent.factory import build_agent
from miniagent.api.settings import ServerSettings
from miniagent.core.exceptions import ConfigurationError
from miniagent.memory.session_store import InMemorySessionStore, SessionStore
from miniagent.memory.sqlite_session_store import SQLiteSessionStore
from miniagent.profile.agent_profile import AgentProfile
from miniagent.workflow.registry import WorkflowRegistry

if TYPE_CHECKING:
    from miniagent.tool.toolset import BaseToolset

# Process-level workflow registry shared across the application.
_WORKFLOW_REGISTRY: WorkflowRegistry = WorkflowRegistry()


def resolve_config_path(config_path: str | None = None) -> Path:
    """Return the configuration file to use.

    Args:
        config_path: Explicit path; when omitted, ``$AGENTS_CONFIG_PATH`` or
            ``agents.yaml`` in the working directory.

    Returns:
        The path to the YAML configuration file.
    """
    if config_path:
        return Path(config_path)
    return ServerSettings.from_environment().agents_config_path


@lru_cache(maxsize=1)
def get_agent_profiles(
    config_path: str | None = None,
) -> list[AgentProfile]:
    """Load and cache agent profiles from the YAML configuration.

    Uses ``AGENTS_CONFIG_PATH`` environment variable when ``config_path``
    is not supplied.

    Args:
        config_path: Optional explicit path to the YAML config file.
            Defaults to ``$AGENTS_CONFIG_PATH`` or ``agents.yaml``.

    Returns:
        A list of validated AgentProfile instances.

    Raises:
        ConfigurationError: If the configuration cannot be loaded.
    """
    return AgentProfile.from_yaml(resolve_config_path(config_path))


def get_agent_by_name(
    agent_name: str,
    config_path: str | None = None,
    toolsets: Sequence[BaseToolset] = (),
) -> BaseAgent:
    """Resolve and instantiate an agent by its name from configuration.

    Args:
        agent_name: The name of the agent as defined in the YAML config.
        config_path: Optional override for the configuration file path.
        toolsets: Connected toolsets the agent's profile may select.

    Returns:
        A fully initialized BaseAgent instance.

    Raises:
        ConfigurationError: If the agent name is not found in config, or its
            profile lists a tool or toolset that is not available.
    """
    profiles = get_agent_profiles(config_path)
    matching = [profile for profile in profiles if profile.name == agent_name]

    if not matching:
        available = [profile.name for profile in profiles]
        raise ConfigurationError(
            f"Agent '{agent_name}' not found in configuration. Available agents: {available}"
        )

    return build_agent(matching[0], toolsets=toolsets)


def create_session_store() -> SessionStore:
    """Create the store that keeps conversations for ``session_id`` requests.

    ``$SESSION_DB_PATH`` selects a SQLite database that survives restarts;
    without it, conversations live in process memory.

    Returns:
        The session store for the application.

    Raises:
        SessionStoreError: If the configured database cannot be opened.
    """
    database_path = ServerSettings.from_environment().session_db_path
    if database_path is not None:
        return SQLiteSessionStore(database_path)
    return InMemorySessionStore()


def get_workflow_registry() -> WorkflowRegistry:
    """Return the process-level workflow registry.

    Returns:
        The shared ``WorkflowRegistry`` instance.
    """
    return _WORKFLOW_REGISTRY


def reset_profile_cache() -> None:
    """Clear the cached agent profiles.

    Useful for testing or when the YAML config changes at runtime.
    """
    get_agent_profiles.cache_clear()
