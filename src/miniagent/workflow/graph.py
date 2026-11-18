"""Graph-based workflow engine for dynamic, stateful execution.

Provides a node-and-edge model where conditional routing determines
the next agent to run based on the current state, enabling
loops, branching, and complex multi-agent orchestration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from miniagent.core.exceptions import WorkflowExecutionError
from miniagent.core.state import State, WorkflowStatus
from miniagent.workflow.base import BaseWorkflow

logger = logging.getLogger(__name__)

# Type alias for edge routing functions.
# Given a State, returns the name of the next node or None to stop.
EdgeRouter = Callable[[State], str | None]

# Sentinel value indicating the workflow should terminate.
END_NODE = "__END__"


class GraphWorkflow(BaseWorkflow):
    """Executes agents in a graph topology with conditional routing.

    Nodes are agents identified by name. Edges define transitions
    between nodes, optionally using router functions to determine
    the next node dynamically based on the current state.

    Attributes:
        edges: Mapping of source node names to their edge routers.
        entry_point: Name of the first node to run.
    """

    def __init__(self, name: str, entry_point: str | None = None) -> None:
        """Initialize the graph workflow.

        Args:
            name: Human-readable workflow identifier.
            entry_point: Name of the starting node.
        """
        super().__init__(name=name)
        self.edges: dict[str, EdgeRouter] = {}
        self.entry_point: str | None = entry_point

    def add_edge(self, source: str, target: str) -> None:
        """Add a deterministic edge between two nodes.

        Args:
            source: The source node name.
            target: The target node name, or ``END_NODE`` to stop.

        Raises:
            ValueError: If an edge from this source already exists.
        """
        if source in self.edges:
            raise ValueError(f"Edge from '{source}' already exists in workflow '{self.name}'.")

        def _fixed_router(_state: State, _target: str = target) -> str:
            return _target

        self.edges[source] = _fixed_router

    def add_conditional_edge(self, source: str, router: EdgeRouter) -> None:
        """Add a conditional edge with dynamic routing.

        The router function receives the current state and returns
        the name of the next node or ``None`` / ``END_NODE`` to stop.

        Args:
            source: The source node name.
            router: A callable that determines the next node.

        Raises:
            ValueError: If an edge from this source already exists.
        """
        if source in self.edges:
            raise ValueError(f"Edge from '{source}' already exists in workflow '{self.name}'.")
        self.edges[source] = router

    async def run(self, state: State | None = None) -> State:
        """Run the graph workflow from the entry point.

        Traverses the graph by executing nodes and following edges
        until ``END_NODE`` is reached or no edge is defined.

        Args:
            state: Optional initial state. Creates a fresh state
                if not provided.

        Returns:
            The final state after the workflow completes.

        Raises:
            WorkflowExecutionError: If the workflow configuration is
                invalid, a node or edge router fails, or the maximum
                step count is exceeded.
        """
        if not self.entry_point:
            raise WorkflowExecutionError(
                f"Graph workflow '{self.name}' has no entry point defined."
            )

        if self.entry_point not in self.agents:
            raise WorkflowExecutionError(
                f"Entry point '{self.entry_point}' is not a registered "
                f"agent in workflow '{self.name}'."
            )

        current_state = state or State()
        current_state = current_state.model_copy(update={"status": WorkflowStatus.RUNNING})
        current_node: str | None = self.entry_point

        while current_node is not None and current_node != END_NODE:
            if current_node not in self.agents:
                raise WorkflowExecutionError(
                    f"Node '{current_node}' is not a registered agent in workflow '{self.name}'."
                )

            if current_state.step_count >= current_state.max_steps:
                logger.warning(
                    "Workflow '%s': max steps (%d) reached at node '%s'.",
                    self.name,
                    current_state.max_steps,
                    current_node,
                )
                current_state = current_state.model_copy(update={"status": WorkflowStatus.FAILED})
                raise WorkflowExecutionError(
                    f"Workflow '{self.name}' exceeded max steps "
                    f"({current_state.max_steps}) at node '{current_node}'."
                )

            agent = self.agents[current_node]
            logger.info(
                "Workflow '%s': executing node '%s' (step %d).",
                self.name,
                current_node,
                current_state.step_count + 1,
            )

            try:
                current_state = current_state.model_copy(update={"current_node": current_node})
                current_state = await agent.run(current_state)
            except WorkflowExecutionError:
                raise
            except Exception as exc:
                current_state = current_state.model_copy(update={"status": WorkflowStatus.FAILED})
                raise WorkflowExecutionError(
                    f"Node '{current_node}' failed in graph workflow '{self.name}'.",
                    cause=exc,
                ) from exc

            # Determine next node via edge routing.
            router = self.edges.get(current_node)
            if router is None:
                logger.info(
                    "Workflow '%s': no edge from '%s', stopping.",
                    self.name,
                    current_node,
                )
                current_node = None
            else:
                try:
                    current_node = router(current_state)
                except Exception as exc:
                    raise WorkflowExecutionError(
                        f"Edge router for node '{current_node}' failed in workflow '{self.name}'.",
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
