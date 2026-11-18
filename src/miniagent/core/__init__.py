"""Shared kernel used by every feature: messages, workflow state, and errors.

Quick import::

    from miniagent.core import Message, MessageRole, State
"""

from miniagent.core.exceptions import (
    AgentExecutionError,
    ConfigurationError,
    LLMProviderError,
    MiniAgentError,
    SessionStoreError,
    ToolInvocationError,
    WorkflowExecutionError,
)
from miniagent.core.message import Message, MessageRole, ToolCall
from miniagent.core.state import State, WorkflowStatus

__all__ = [
    "AgentExecutionError",
    "ConfigurationError",
    "LLMProviderError",
    "Message",
    "MessageRole",
    "MiniAgentError",
    "SessionStoreError",
    "State",
    "ToolCall",
    "ToolInvocationError",
    "WorkflowExecutionError",
    "WorkflowStatus",
]
