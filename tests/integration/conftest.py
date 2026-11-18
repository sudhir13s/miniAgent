"""Shared fixtures for Phase-4 integration tests.

All fixtures are designed to avoid live API calls:
- ``mock_litellm`` patches ``litellm.acompletion`` with deterministic responses.
- ``base_state`` provides a State seeded with one user message.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from miniagent.agent.agent import Agent
from miniagent.core.message import Message, MessageRole
from miniagent.core.state import State
from miniagent.llm.litellm_provider import LiteLLMProvider
from miniagent.profile.agent_profile import AgentProfile

# ---------------------------------------------------------------------------
# Helper builders (reused by fixtures and tests alike)
# ---------------------------------------------------------------------------


def make_litellm_response(content: str) -> MagicMock:
    """Build a minimal mock litellm response object."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_state() -> State:
    """A State seeded with a single user message."""
    return State(
        messages=[Message(role=MessageRole.USER, content="Start the workflow.")],
        max_steps=500,
    )


@pytest.fixture()
def mock_litellm() -> Any:
    """Patch litellm.acompletion to return a deterministic response.

    Individual tests can override ``mock_litellm.return_value`` or
    ``mock_litellm.side_effect`` as needed.
    """
    target = "miniagent.llm.litellm_provider.litellm.acompletion"
    with patch(target, new_callable=AsyncMock) as mock_fn:
        mock_fn.return_value = make_litellm_response("Default LiteLLM response.")
        yield mock_fn


# ---------------------------------------------------------------------------
# Agent factory helpers (used across test modules)
# ---------------------------------------------------------------------------


def make_agent(
    profile: AgentProfile,
    tools: dict[str, Any] | None = None,
) -> Agent:
    """Create an Agent using the real LiteLLMProvider (which will be mocked)."""
    return Agent(profile, LiteLLMProvider(), tools)
