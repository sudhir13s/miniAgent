"""Abstract base class defining the contract for all miniAgent agents.

``Agent`` is the built-in implementation. A custom agent implements ``run``
(and optionally ``stream``) to plug into any workflow or the REST API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from miniagent.agent.events import AgentEvent, RunCompleted
from miniagent.core.state import State
from miniagent.profile.agent_profile import AgentProfile


class BaseAgent(ABC):
    """Abstract base class for all agents in the miniAgent framework.

    Agents are autonomous units that process messages, optionally invoke
    tools, and produce responses. Each agent is configured via an
    ``AgentProfile`` that specifies its persona, model, and tools.

    Attributes:
        profile: The declarative configuration for this agent.
    """

    def __init__(self, profile: AgentProfile) -> None:
        """Initialize the agent with its profile.

        Args:
            profile: The agent's declarative configuration, typically
                loaded from a YAML file.
        """
        self.profile = profile

    @abstractmethod
    async def run(self, state: State) -> State:
        """Run the agent's reasoning loop on the given state.

        The agent reads messages from the state, interacts with its
        configured LLM, optionally invokes tools, and returns an
        updated state with new messages appended.

        Args:
            state: The current workflow state containing conversation
                history and variables.

        Returns:
            An updated State with new messages and possibly modified
            variables.

        Raises:
            AgentExecutionError: If the agent encounters an
                unrecoverable error during execution.
        """

    async def stream(self, state: State) -> AsyncIterator[AgentEvent]:
        """Run the agent and yield its progress as events.

        This default does not stream: it calls ``run()`` and yields one
        ``RunCompleted``. Agents that can report progress should override it.

        Args:
            state: The current workflow state.

        Yields:
            ``AgentEvent`` objects, ending with one ``RunCompleted``.

        Raises:
            AgentExecutionError: If the run fails.
        """
        yield RunCompleted(await self.run(state))
