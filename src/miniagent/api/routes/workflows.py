"""FastAPI routes for workflow management and execution.

Exposes RESTful endpoints for listing registered workflows, running a
workflow synchronously, and streaming workflow execution progress via
Server-Sent Events (SSE).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from miniagent.api.dependencies import get_workflow_registry
from miniagent.api.sse import format_sse, sse_response
from miniagent.api.stream_events import (
    StreamEventName,
    WorkflowErrorPayload,
    WorkflowStartedPayload,
)
from miniagent.core.exceptions import WorkflowExecutionError
from miniagent.core.message import Message, MessageRole
from miniagent.core.state import State

router = APIRouter(prefix="/workflows", tags=["Workflows"])


# ── Response models ───────────────────────────────────────────────────────────


class WorkflowSummary(BaseModel):
    """Brief description of a registered workflow.

    Attributes:
        name: Unique workflow identifier.
    """

    name: str


class WorkflowRunRequest(BaseModel):
    """Request body for workflow execution.

    Attributes:
        message: Optional user message to seed the initial state.
        variables: Optional state variables.
        max_steps: Maximum execution steps allowed.
    """

    message: str | None = Field(
        default=None, description="Optional user message to seed the workflow."
    )
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional state variables.",
    )
    max_steps: int = Field(default=100, ge=1, le=1000, description="Maximum execution steps.")


class WorkflowRunResponse(BaseModel):
    """Response from workflow execution.

    Attributes:
        workflow_name: Name of the executed workflow.
        status: Final workflow status.
        step_count: Total steps executed.
        variables: Final state variables.
        message_count: Total messages in the final state.
    """

    workflow_name: str
    status: str
    step_count: int
    variables: dict[str, Any]
    message_count: int


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_initial_state(request: WorkflowRunRequest) -> State:
    """Build the initial State from a run request.

    Args:
        request: The workflow run request.

    Returns:
        A State seeded with the request's message and variables.
    """
    state = State(variables=request.variables, max_steps=request.max_steps)
    if request.message:
        state = state.add_message(Message(role=MessageRole.USER, content=request.message))
    return state


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get(
    "",
    response_model=list[WorkflowSummary],
    summary="List all workflows",
    description="Returns the names of every registered workflow.",
)
def list_workflows() -> list[WorkflowSummary]:
    """List all registered workflows.

    Returns:
        A list of workflow summaries.
    """
    registry = get_workflow_registry()
    return [WorkflowSummary(name=n) for n in registry.list_workflows()]


@router.post(
    "/{workflow_name}/run",
    response_model=WorkflowRunResponse,
    summary="Run a workflow",
    description="Run a registered workflow synchronously and return the final state.",
)
async def run_workflow(
    workflow_name: str,
    request: WorkflowRunRequest,
) -> WorkflowRunResponse:
    """Run a registered workflow synchronously.

    Args:
        workflow_name: The workflow identifier from the URL path.
        request: Run request with optional message, variables, and max_steps.

    Returns:
        Final state metadata after execution completes.

    Raises:
        HTTPException: 404 if the workflow is not found, 500 on failure.
    """
    registry = get_workflow_registry()
    try:
        workflow = registry.get(workflow_name)
    except WorkflowExecutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    initial_state = _build_initial_state(request)

    try:
        final_state = await workflow.run(initial_state)
    except WorkflowExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _run_response(workflow_name, final_state)


@router.post(
    "/{workflow_name}/run/stream",
    summary="Stream workflow execution",
    description=(
        "Run a workflow and stream progress as Server-Sent Events. "
        "Emits ``started``, ``completed``, and ``error`` events."
    ),
    response_class=StreamingResponse,
)
async def stream_workflow(
    workflow_name: str,
    request: WorkflowRunRequest,
) -> StreamingResponse:
    """Stream workflow execution progress via Server-Sent Events.

    Args:
        workflow_name: The workflow identifier from the URL path.
        request: Run request with optional message, variables, and max_steps.

    Returns:
        A ``StreamingResponse`` with ``text/event-stream`` content type.

    Raises:
        HTTPException: 404 if the workflow is not found.
    """
    registry = get_workflow_registry()
    try:
        workflow = registry.get(workflow_name)
    except WorkflowExecutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    initial_state = _build_initial_state(request)

    async def _event_stream() -> AsyncIterator[str]:
        started = WorkflowStartedPayload(workflow_name=workflow_name)
        yield format_sse(StreamEventName.STARTED, started)
        try:
            final_state = await workflow.run(initial_state)
            yield format_sse(StreamEventName.COMPLETED, _run_response(workflow_name, final_state))
        except Exception as exc:
            failure = WorkflowErrorPayload(workflow_name=workflow_name, detail=str(exc))
            yield format_sse(StreamEventName.ERROR, failure)

    return sse_response(_event_stream())


def _run_response(workflow_name: str, final_state: State) -> WorkflowRunResponse:
    """Describe a finished run; shared by ``/run`` and the stream's ``completed`` event."""
    return WorkflowRunResponse(
        workflow_name=workflow_name,
        status=final_state.status.value,
        step_count=final_state.step_count,
        variables=dict(final_state.variables),
        message_count=len(final_state.messages),
    )
