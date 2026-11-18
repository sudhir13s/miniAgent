"""Run state: the messages, variables and step counters a run carries.

``State`` is the input and output of every agent run and every workflow.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from miniagent.core.message import Message


class WorkflowStatus(str, Enum):
    """Status of a workflow execution.

    Attributes:
        PENDING: Workflow has been created but not yet started.
        RUNNING: Workflow is currently executing.
        COMPLETED: Workflow finished successfully.
        FAILED: Workflow terminated due to an error.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class State(BaseModel):
    """Runtime state container for an agent or workflow run.

    Holds the context that flows through an agent run, the nodes of a
    workflow graph or the steps of a pipeline. Its helper methods never
    change it in place; each returns a new ``State``.

    Attributes:
        id: Unique identifier for this state instance.
        status: Current execution status of the workflow.
        messages: Ordered list of messages accumulated during execution.
        variables: Arbitrary key-value store for intermediate data.
        current_node: Name of the node currently being executed.
        step_count: Number of steps executed so far.
        max_steps: Maximum allowed steps before forced termination.
        created_at: UTC timestamp when the state was initialized.
        updated_at: UTC timestamp of the last state mutation.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this state instance.",
    )
    status: WorkflowStatus = Field(
        default=WorkflowStatus.PENDING,
        description="Current execution status.",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="Ordered list of messages accumulated during execution.",
    )
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value store for intermediate data.",
    )
    current_node: str | None = Field(
        default=None,
        description="Name of the node currently being executed.",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps executed so far.",
    )
    max_steps: int = Field(
        default=100,
        ge=1,
        description="Maximum allowed steps before forced termination.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the state was initialized.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the last state mutation.",
    )

    def add_message(self, message: Message) -> State:
        """Create a new state with the given message appended.

        Args:
            message: The message to append to the conversation history.

        Returns:
            A new State instance with the message added and timestamp updated.
        """
        return self.model_copy(
            update={
                "messages": [*self.messages, message],
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def increment_step(self) -> State:
        """Create a new state with step_count incremented by one.

        Returns:
            A new State instance with step_count incremented.

        Raises:
            ValueError: If step_count has reached max_steps.
        """
        if self.step_count >= self.max_steps:
            raise ValueError(
                f"Step count ({self.step_count}) has reached "
                f"maximum allowed steps ({self.max_steps})."
            )
        return self.model_copy(
            update={
                "step_count": self.step_count + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def set_variable(self, key: str, value: Any) -> State:
        """Create a new state with a variable set or updated.

        Args:
            key: The variable name.
            value: The variable value.

        Returns:
            A new State instance with the variable set.
        """
        new_vars = {**self.variables, key: value}
        return self.model_copy(
            update={
                "variables": new_vars,
                "updated_at": datetime.now(timezone.utc),
            }
        )
