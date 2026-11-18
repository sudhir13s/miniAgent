"""Workflow registry — named workflow instance management.

Provides a simple in-process store for named workflow instances so that
the API layer and other consumers can resolve a workflow by name without
holding direct references.

Usage::

    from miniagent.workflow.registry import WorkflowRegistry

    registry = WorkflowRegistry()
    registry.register("pipeline", my_sequential_workflow)

    workflow = registry.get("pipeline")
    state = await workflow.run(initial_state)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from miniagent.core.exceptions import WorkflowExecutionError

if TYPE_CHECKING:
    from miniagent.workflow.base import BaseWorkflow


class WorkflowRegistry:
    """In-process store for named workflow instances.

    Attributes:
        _workflows: Mapping of workflow name → ``BaseWorkflow`` instance.
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._workflows: dict[str, BaseWorkflow] = {}

    def register(self, name: str, workflow: BaseWorkflow) -> None:
        """Register a workflow under the given name.

        Args:
            name: Unique name to register the workflow under.
            workflow: The workflow instance to store.

        Raises:
            ValueError: If a workflow with the same name is already registered.
        """
        if name in self._workflows:
            raise ValueError(
                f"A workflow named '{name}' is already registered. "
                "Use a unique name or call unregister() first."
            )
        self._workflows[name] = workflow

    def get(self, name: str) -> BaseWorkflow:
        """Retrieve a workflow by name.

        Args:
            name: The workflow name.

        Returns:
            The registered ``BaseWorkflow`` instance.

        Raises:
            WorkflowExecutionError: If no workflow is registered under ``name``.
        """
        workflow = self._workflows.get(name)
        if workflow is None:
            available = ", ".join(sorted(self._workflows)) or "none"
            raise WorkflowExecutionError(
                f"No workflow registered under '{name}'. Available workflows: {available}."
            )
        return workflow

    def unregister(self, name: str) -> None:
        """Remove a workflow from the registry.

        Args:
            name: The workflow name to remove.

        Raises:
            WorkflowExecutionError: If no workflow is registered under ``name``.
        """
        if name not in self._workflows:
            raise WorkflowExecutionError(f"Cannot unregister '{name}': not registered.")
        del self._workflows[name]

    def list_workflows(self) -> list[str]:
        """Return the names of all registered workflows.

        Returns:
            Sorted list of registered workflow names.
        """
        return sorted(self._workflows.keys())

    def __len__(self) -> int:
        """Return the number of registered workflows."""
        return len(self._workflows)

    def __contains__(self, name: object) -> bool:
        """Support ``name in registry`` checks."""
        return name in self._workflows
