"""miniAgent: a lightweight agent SDK — one agent loop, any model via LiteLLM.

Quick import::

    from miniagent import AgentProfile, build_agent

    agent = build_agent(AgentProfile.from_yaml("agents.yaml", name="assistant"))
"""

from miniagent.agent import Agent, BaseAgent, build_agent
from miniagent.core import Message, MessageRole, State
from miniagent.profile import AgentProfile, ModelConfig, ProfileTemplate

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentProfile",
    "BaseAgent",
    "Message",
    "MessageRole",
    "ModelConfig",
    "ProfileTemplate",
    "State",
    "__version__",
    "build_agent",
]
