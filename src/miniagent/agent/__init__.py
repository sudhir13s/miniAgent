"""Agents: the ``BaseAgent`` contract, the built-in ``Agent``, and ``build_agent``.

Quick import::

    from miniagent.agent import Agent, BaseAgent, build_agent
    from miniagent.agent import RunCompleted, TextDelta  # agent.stream() events
"""

from miniagent.agent.agent import Agent
from miniagent.agent.base import BaseAgent
from miniagent.agent.events import (
    AgentEvent,
    RunCompleted,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from miniagent.agent.factory import build_agent
from miniagent.agent.structured_output import OUTPUT_VARIABLE

__all__ = [
    "OUTPUT_VARIABLE",
    "Agent",
    "AgentEvent",
    "BaseAgent",
    "RunCompleted",
    "TextDelta",
    "ToolCallFinished",
    "ToolCallStarted",
    "build_agent",
]
