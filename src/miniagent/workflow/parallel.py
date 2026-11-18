"""ParallelWorkflow — fan-out / fan-in concurrent agent execution.

Runs multiple agents concurrently using ``asyncio.gather`` (fan-out), then
merges all resulting states into a single final ``State`` (fan-in).

Fan-in strategy:
- Messages the branches added follow the initial ones, sorted by timestamp.
- Variables from all branches are merged (later agents in ``agent_order``
  overwrite earlier ones on key collision).
- ``step_count`` is the starting count plus the steps each branch added.
- Final status is ``COMPLETED`` unless any branch is ``FAILED``.

Usage::

    workflow = ParallelWorkflow("research-pipeline")
    workflow.register_agent(agent_a)
    workflow.register_agent(agent_b)
    workflow.register_agent(agent_c)

    final_state = await workflow.run(initial_state)
    # final_state.messages contains all messages from all three agents
"""

from __future__ import annotations

import asyncio
import logging

from miniagent.core.exceptions import WorkflowExecutionError
from miniagent.core.state import State, WorkflowStatus
from miniagent.workflow.base import BaseWorkflow

logger = logging.getLogger(__name__)


class ParallelWorkflow(BaseWorkflow):
    """Executes agents concurrently and merges their outputs.

    Each registered agent runs independently from the same initial state.
    After all agents complete, their states are merged into one final state.

    Attributes:
        agent_order: Ordered list of agent names defining the execution set.
            If empty, all registered agents are run.
    """

    def __init__(
        self,
        name: str,
        agent_order: list[str] | None = None,
    ) -> None:
        """Initialise the parallel workflow.

        Args:
            name: Human-readable workflow identifier.
            agent_order: Optional explicit list of agent names to run.
                If not provided, all registered agents are run.
        """
        super().__init__(name=name)
        self.agent_order: list[str] = agent_order or []

    async def run(self, state: State | None = None) -> State:
        """Run all agents concurrently and merge results.

        Args:
            state: Optional initial state. Creates a fresh ``State`` if
                not provided. Every agent branch starts from this
                initial state.

        Returns:
            Merged ``State``: the initial messages, then the messages every
            branch added sorted by timestamp, and the merged variables.

        Raises:
            WorkflowExecutionError: If no agents are registered or if a
                concurrent branch raises an unexpected error.
        """
        order = self._agents_to_run(self.agent_order)
        initial = state or State()
        running = initial.model_copy(update={"status": WorkflowStatus.RUNNING})

        logger.info(
            "ParallelWorkflow '%s': launching %d agents concurrently",
            self.name,
            len(order),
        )

        try:
            branch_states: list[State] = await asyncio.gather(
                *[self.agents[name].run(running) for name in order],
                return_exceptions=False,
            )
        except WorkflowExecutionError:
            raise
        except Exception as exc:
            raise WorkflowExecutionError(
                f"ParallelWorkflow '{self.name}' gather failed: {exc}", cause=exc
            ) from exc

        final = self._merge_states(running, branch_states)
        logger.info(
            "ParallelWorkflow '%s': merged %d branches → %d messages",
            self.name,
            len(branch_states),
            len(final.messages),
        )
        return final

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _merge_states(self, base: State, branches: list[State]) -> State:
        """Merge branch states into a single state.

        Args:
            base: The initial state before any branch ran.
            branches: Results from each concurrent agent branch.

        Returns:
            A single merged ``State``.
        """
        # Collect all NEW messages (not in base) from every branch
        base_ids = {message.id for message in base.messages}
        new_messages = [
            message
            for branch in branches
            for message in branch.messages
            if message.id not in base_ids
        ]
        # Sort by timestamp so merged history is chronologically coherent
        new_messages.sort(key=lambda message: message.timestamp)

        # Merge variables (last-write wins, ordered by agent_order)
        merged_vars = dict(base.variables)
        for branch in branches:
            merged_vars.update(branch.variables)

        # Every branch starts from the base count, so add only the steps each one took.
        total_steps = base.step_count + sum(b.step_count - base.step_count for b in branches)

        # Status: FAILED if any branch failed, otherwise COMPLETED
        any_failed = any(b.status == WorkflowStatus.FAILED for b in branches)
        final_status = WorkflowStatus.FAILED if any_failed else WorkflowStatus.COMPLETED

        return base.model_copy(
            update={
                "messages": [*base.messages, *new_messages],
                "variables": merged_vars,
                "step_count": total_steps,
                "status": final_status,
                "current_node": None,
            }
        )
