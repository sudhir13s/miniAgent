"""Build a ready-to-run ``Agent`` from an ``AgentProfile``.

Resolves the tool and toolset names listed on the profile to instances and
attaches the LLM provider, so callers holding only a profile (e.g. one
loaded from YAML) can get a working agent in one call::

    profile = AgentProfile.from_yaml("agents.yaml", name="researcher")
    agent = build_agent(profile)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from miniagent.agent.agent import Agent
from miniagent.core.exceptions import ConfigurationError
from miniagent.llm.registry import get_default_provider
from miniagent.memory.context_window import TokenBudgetPolicy
from miniagent.tool.catalog import builtin_tools

if TYPE_CHECKING:
    from pydantic import BaseModel

    from miniagent.llm.base import BaseLLMProvider
    from miniagent.memory.context_window import ContextPolicy
    from miniagent.profile.agent_profile import AgentProfile
    from miniagent.tool.base import BaseTool
    from miniagent.tool.toolset import BaseToolset


def build_agent(
    profile: AgentProfile,
    *,
    tools: Mapping[str, BaseTool] | None = None,
    toolsets: Sequence[BaseToolset] = (),
    llm_provider: BaseLLMProvider | None = None,
    output_type: type[BaseModel] | None = None,
) -> Agent:
    """Create an ``Agent`` for a profile, wiring its tools, toolsets and LLM provider.

    Each name in ``profile.tools`` is looked up first in ``tools`` and then
    in the built-in tools, so custom tools can add to or replace built-ins.
    Each name in ``profile.toolsets`` selects one of ``toolsets`` by name.

    Args:
        profile: The agent's declarative configuration.
        tools: Custom tools by name, available in addition to the built-ins.
        toolsets: Toolsets the profile may select (e.g. connected MCP servers).
        llm_provider: Provider to use; defaults to the process-wide default
            (``LiteLLMProvider`` unless overridden).
        output_type: Pydantic model the final answer must match; see
            ``miniagent.agent.structured_output``.

    Returns:
        An ``Agent`` with exactly the tools and toolsets its profile lists,
        trimming its context to ``profile.max_context_tokens`` when set.

    Raises:
        ConfigurationError: If the profile lists a tool or toolset that is not
            available, or two toolsets share a name.
    """
    return Agent(
        profile=profile,
        llm_provider=llm_provider or get_default_provider(),
        tools=_select_tools(profile, {**builtin_tools(), **(tools or {})}),
        toolsets=_select_toolsets(profile, toolsets),
        context_policy=_context_policy(profile),
        output_type=output_type,
    )


def _context_policy(profile: AgentProfile) -> ContextPolicy | None:
    """Return a token budget policy when the profile sets a budget."""
    if profile.max_context_tokens is None:
        return None
    return TokenBudgetPolicy(profile.max_context_tokens)


def _select_tools(profile: AgentProfile, available: Mapping[str, BaseTool]) -> dict[str, BaseTool]:
    """Return the tools ``profile.tools`` names, failing on any that are missing."""
    _reject_unknown(profile, "tools", profile.tools, available)
    return {name: available[name] for name in profile.tools}


def _select_toolsets(profile: AgentProfile, toolsets: Sequence[BaseToolset]) -> list[BaseToolset]:
    """Return the toolsets ``profile.toolsets`` names, failing on missing or ambiguous ones."""
    by_name: dict[str, BaseToolset] = {}
    for toolset in toolsets:
        if toolset.name in by_name:
            raise ConfigurationError(f"There is more than one toolset named '{toolset.name}'.")
        by_name[toolset.name] = toolset
    _reject_unknown(profile, "toolsets", profile.toolsets, by_name)
    return [by_name[name] for name in profile.toolsets]


def _reject_unknown(
    profile: AgentProfile, kind: str, requested: Sequence[str], available: Mapping[str, object]
) -> None:
    """Raise if ``requested`` names anything ``available`` lacks."""
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ConfigurationError(
            f"Agent '{profile.name}' lists unknown {kind}: {', '.join(unknown)}. "
            f"Available {kind}: {', '.join(sorted(available)) or 'none'}."
        )
