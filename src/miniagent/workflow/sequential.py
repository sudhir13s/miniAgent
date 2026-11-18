"""Sequential workflow engine for step-by-step pipeline execution.

Executes a predefined ordered list of agents one after another,
passing state through each agent in sequence.
"""

from __future__ import annotations

import logging

from miniagent.core.exceptions import WorkflowExecutionError
from miniagent.core.state import State, WorkflowStatus
from miniagent.workflow.base import BaseWorkflow

logger = logging.getLogger(__name__)


class SequentialWorkflow(BaseWorkflow):
    """Executes agents in a fixed, predefined order.

    Each agent receives the state output of the previous agent,
    forming a linear pipeline. Execution halts on error or when
    all agents have been processed.

    Attributes:
        agent_order: Ordered list of agent names defining the
            execution sequence.
    """

    def __init__(
        self,
        name: str,
        agent_order: list[str] | None = None,
    ) -> None:
        """Initialize the sequential workflow.

        Args:
            name: Human-readable workflow identifier.
            agent_order: Ordered list of agent names. If not provided,
                agents are executed in registration order.
        """
        super().__init__(name=name)
        self.agent_order: list[str] = agent_order or []

    async def run(self, state: State | None = None) -> State:
        """Run agents sequentially in the defined order.

        Args:
            state: Optional initial state. Creates a fresh state
                if not provided.

        Returns:
            The final state after all agents have executed.

        Raises:
            WorkflowExecutionError: If any agent fails or if the
                workflow configuration is invalid.
        """
        order = self._agents_to_run(self.agent_order)
        current_state = state or State()
        current_state = current_state.model_copy(update={"status": WorkflowStatus.RUNNING})

        for position, agent_name in enumerate(order, start=1):
            agent = self.agents[agent_name]
            logger.info(
                "Workflow '%s': executing agent '%s' (step %d/%d)",
                self.name,
                agent_name,
                position,
                len(order),
            )

            try:
                current_state = current_state.model_copy(update={"current_node": agent_name})
                current_state = await agent.run(current_state)
            except Exception as exc:
                current_state = current_state.model_copy(update={"status": WorkflowStatus.FAILED})
                raise WorkflowExecutionError(
                    f"Agent '{agent_name}' failed in sequential workflow '{self.name}'.",
                    cause=exc,
                ) from exc

        current_state = current_state.model_copy(
            update={
                "status": WorkflowStatus.COMPLETED,
                "current_node": None,
            }
        )
        logger.info("Workflow '%s': completed successfully.", self.name)
        return current_state
