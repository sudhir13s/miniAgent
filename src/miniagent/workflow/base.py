"""Abstract base class defining the contract for miniAgent workflows.

Workflows orchestrate the execution of agents in structured patterns such
as sequential pipelines or graph-based state machines. Tools belong to the
agents themselves (see ``build_agent``), not to the workflow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from miniagent.agent.base import BaseAgent
from miniagent.core.exceptions import WorkflowExecutionError
from miniagent.core.state import State


class BaseWorkflow(ABC):
    """Abstract base class for all workflows in the miniAgent framework.

    A workflow defines the execution strategy for one or more agents and
    manages the state passed between them.

    Attributes:
        name: Human-readable identifier for this workflow.
        agents: Mapping of agent names to their instances.
    """

    def __init__(
        self,
        name: str,
        agents: dict[str, BaseAgent] | None = None,
    ) -> None:
        """Initialize the workflow.

        Args:
            name: Human-readable identifier for this workflow.
            agents: Mapping of agent names to their instances.
        """
        self.name = name
        self.agents: dict[str, BaseAgent] = agents or {}

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with this workflow.

        Args:
            agent: The agent instance to register.

        Raises:
            ValueError: If an agent with the same name is already
                registered.
        """
        if agent.profile.name in self.agents:
            raise ValueError(
                f"Agent '{agent.profile.name}' is already registered in workflow '{self.name}'."
            )
        self.agents[agent.profile.name] = agent

    def _agents_to_run(self, agent_order: list[str]) -> list[str]:
        """Return the agent names to run, checked before anything runs.

        Args:
            agent_order: The configured order; empty means registration order.

        Returns:
            The names in execution order.

        Raises:
            WorkflowExecutionError: If there are no agents to run, or the
                order names an agent that is not registered.
        """
        order = agent_order or list(self.agents)
        if not order:
            raise WorkflowExecutionError(
                f"Workflow '{self.name}' has no agents registered. "
                "Call register_agent() before run()."
            )
        for agent_name in order:
            if agent_name not in self.agents:
                raise WorkflowExecutionError(
                    f"Agent '{agent_name}' is not registered in workflow '{self.name}'. "
                    f"Registered: {list(self.agents)}"
                )
        return order

    @abstractmethod
    async def run(self, state: State | None = None) -> State:
        """Run the workflow from start to completion.

        Args:
            state: Optional initial state. If not provided, a fresh
                state will be created.

        Returns:
            The final state after the workflow completes.

        Raises:
            WorkflowExecutionError: If the workflow encounters an
                unrecoverable error.
        """
