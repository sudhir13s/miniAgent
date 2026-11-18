"""LoopWorkflow — iterative agent execution with exit-condition support.

Runs a single agent (or a set of agents sequentially) in a loop, stopping
when an ``ExitCondition`` callable returns ``True`` or when ``max_iterations``
is reached.

This is useful for iterative refinement, polling, or self-correction
patterns where an agent needs to keep running until a quality threshold
is satisfied.

Usage::

    from miniagent.workflow.loop import LoopWorkflow, ExitCondition

    # Stop when the state has >= 3 AI messages
    def enough_responses(state: State) -> bool:
        return sum(1 for message in state.messages if message.role == MessageRole.AI) >= 3

    workflow = LoopWorkflow("refine", exit_condition=enough_responses, max_iterations=10)
    workflow.register_agent(my_agent)
    final_state = await workflow.run(initial_state)
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from miniagent.core.exceptions import WorkflowExecutionError
from miniagent.core.state import State, WorkflowStatus
from miniagent.workflow.base import BaseWorkflow

logger = logging.getLogger(__name__)

# Type alias: a callable that takes the current State and returns True to stop
ExitCondition = Callable[[State], bool]


class LoopWorkflow(BaseWorkflow):
    """Runs an agent in a loop until an exit condition is met.

    On each iteration, agents are executed in ``agent_order`` (or
    registration order) sequentially, forming one "iteration unit".
    After each iteration the ``exit_condition`` is evaluated; if it
    returns ``True`` the loop ends.

    If no ``exit_condition`` is provided the loop runs until
    ``max_iterations`` is exhausted.

    Attributes:
        agent_order: Ordered list of agent names per iteration.
        exit_condition: Optional callable ``(State) -> bool`` that stops
            the loop when it returns ``True``.
        max_iterations: Hard ceiling on loop count.
    """

    def __init__(
        self,
        name: str,
        agent_order: list[str] | None = None,
        exit_condition: ExitCondition | None = None,
        max_iterations: int = 10,
    ) -> None:
        """Initialise the loop workflow.

        Args:
            name: Human-readable workflow identifier.
            agent_order: Ordered list of agent names to run per
                iteration. Defaults to registration order.
            exit_condition: Optional ``(State) -> bool`` callable. The
                loop stops when it returns ``True``.
            max_iterations: Maximum number of loop iterations before
                forced termination. Defaults to 10.

        Raises:
            ValueError: If ``max_iterations`` is less than 1.
        """
        super().__init__(name=name)
        self.agent_order: list[str] = agent_order or []
        self.exit_condition: ExitCondition | None = exit_condition
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        self.max_iterations: int = max_iterations

    async def run(self, state: State | None = None) -> State:
        """Run the iterative loop.

        Args:
            state: Optional initial state. A fresh ``State`` is created
                if not provided.

        Returns:
            The state after the exit condition is satisfied or
            ``max_iterations`` is reached.

        Raises:
            WorkflowExecutionError: If no agents are registered, if a
                referenced agent is not found, or if an agent raises.
        """
        order = self._agents_to_run(self.agent_order)
        current = state or State()
        current = current.model_copy(update={"status": WorkflowStatus.RUNNING})

        logger.info(
            "LoopWorkflow '%s': starting (agents=%s, max_iter=%d)",
            self.name,
            order,
            self.max_iterations,
        )

        for iteration in range(self.max_iterations):
            logger.debug(
                "LoopWorkflow '%s': iteration %d/%d",
                self.name,
                iteration + 1,
                self.max_iterations,
            )

            # Run all agents sequentially within this iteration
            for agent_name in order:
                agent = self.agents[agent_name]
                try:
                    current = current.model_copy(update={"current_node": agent_name})
                    current = await agent.run(current)
                except Exception as exc:
                    current = current.model_copy(update={"status": WorkflowStatus.FAILED})
                    raise WorkflowExecutionError(
                        f"Agent '{agent_name}' failed in LoopWorkflow "
                        f"'{self.name}' at iteration {iteration + 1}.",
                        cause=exc,
                    ) from exc

            # Evaluate exit condition after full iteration
            if self.exit_condition is not None and self.exit_condition(current):
                logger.info(
                    "LoopWorkflow '%s': exit condition satisfied at iteration %d.",
                    self.name,
                    iteration + 1,
                )
                break
        else:
            logger.warning(
                "LoopWorkflow '%s': reached max_iterations=%d without exit condition.",
                self.name,
                self.max_iterations,
            )

        current = current.model_copy(
            update={
                "status": WorkflowStatus.COMPLETED,
                "current_node": None,
            }
        )
        logger.info("LoopWorkflow '%s': completed.", self.name)
        return current
