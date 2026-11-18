"""Deterministic fake agents and profiles for workflow unit tests.

No LLM is involved: each fake implements ``BaseAgent`` with fixed behavior
so workflow tests exercise orchestration logic only.
"""

from __future__ import annotations

from miniagent.agent.base import BaseAgent
from miniagent.core.message import Message, MessageRole
from miniagent.core.state import State
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.model_config import ModelConfig

FAKE_MODEL = "openai/gpt-4o"


def make_profile(name: str = "agent") -> AgentProfile:
    """Build a minimal valid profile for a fake agent."""
    return AgentProfile(name=name, model=ModelConfig(id=FAKE_MODEL))


class EchoAgent(BaseAgent):
    """Appends ``Echo from <name>`` and increments the step count."""

    async def run(self, state: State) -> State:
        message = Message(role=MessageRole.AI, content=f"Echo from {self.profile.name}")
        return state.add_message(message).increment_step()


class FailingAgent(BaseAgent):
    """Always raises, to test workflow error propagation."""

    async def run(self, state: State) -> State:
        raise RuntimeError("Intentional failure")


class AppendAgent(BaseAgent):
    """Appends ``response-<label>`` to state."""

    def __init__(self, profile: AgentProfile, label: str) -> None:
        super().__init__(profile)
        self.label = label

    async def run(self, state: State) -> State:
        message = Message(role=MessageRole.AI, content=f"response-{self.label}")
        return state.add_message(message).increment_step()


class CounterAgent(BaseAgent):
    """Increments the ``count`` state variable on each execution."""

    async def run(self, state: State) -> State:
        count = state.variables.get("count", 0) + 1
        message = Message(role=MessageRole.AI, content=f"count={count}")
        return state.add_message(message).increment_step().set_variable("count", count)
