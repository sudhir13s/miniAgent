"""The streaming event schema: every Server-Sent Event the API sends.

Each event has a name (``StreamEventName``) and a payload model. The
endpoints build these models and ``sse.format_sse`` writes them, so an event's
name and fields are stated in this file and nowhere else. The one exception
is ``completed``, whose payload is the endpoint's own run response model.

Agent stream (``POST /agents/{name}/run/stream``):
    ``text`` · ``tool_call`` · ``tool_result`` · then ``completed``
    (an ``AgentRunResponse``) or ``error``.

Workflow stream (``POST /workflows/{name}/run/stream``):
    ``started`` · then ``completed`` (a ``WorkflowRunResponse``) or ``error``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class StreamEventName(str, Enum):
    """The ``event:`` names a stream can carry."""

    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STARTED = "started"
    COMPLETED = "completed"
    ERROR = "error"


class TextPayload(BaseModel):
    """``text``: the model wrote more of its reply.

    Attributes:
        delta: The new text; join every delta to get the full reply.
    """

    delta: str


class ToolCallPayload(BaseModel):
    """``tool_call``: a tool is about to run.

    Attributes:
        id: Identifier linking this call to its ``tool_result``.
        name: The tool's name.
        arguments: The arguments the model passed.
    """

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResultPayload(BaseModel):
    """``tool_result``: a tool finished.

    Attributes:
        id: Identifier of the ``tool_call`` this answers.
        name: The tool's name.
        content: What the tool returned, or its error message.
        is_error: Whether the tool failed.
    """

    id: str | None
    name: str | None
    content: str
    is_error: bool


class WorkflowStartedPayload(BaseModel):
    """``started``: a workflow run began.

    Attributes:
        workflow_name: The workflow that is running.
    """

    workflow_name: str


class ErrorPayload(BaseModel):
    """``error``: an agent run failed; no further events follow.

    Attributes:
        detail: What went wrong.
    """

    detail: str


class WorkflowErrorPayload(ErrorPayload):
    """``error``: a workflow run failed; no further events follow.

    Attributes:
        workflow_name: The workflow that failed.
    """

    workflow_name: str
