"""FastAPI routes for agent management and execution.

Exposes RESTful endpoints for listing agent profiles, retrieving a
single agent's configuration, and running an agent on a user message,
optionally continuing a stored conversation, with or without streaming.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from miniagent.agent.base import BaseAgent
from miniagent.agent.events import RunCompleted
from miniagent.api.agent_runs import RunSession, final_response, progress_event
from miniagent.api.dependencies import get_agent_by_name, get_agent_profiles
from miniagent.api.sse import format_sse, sse_response
from miniagent.api.stream_events import ErrorPayload, StreamEventName
from miniagent.core.exceptions import (
    AgentExecutionError,
    ConfigurationError,
    SessionStoreError,
)
from miniagent.core.state import State
from miniagent.memory.session_store import MAX_SESSION_ID_LENGTH, SESSION_ID_PATTERN

router = APIRouter(prefix="/agents", tags=["Agents"])


# ── Response models ───────────────────────────────────────────────────────────


class AgentSummary(BaseModel):
    """Brief description of a registered agent.

    Attributes:
        name: Unique agent identifier.
        description: Human-readable purpose of the agent.
        model: LiteLLM model string, e.g. ``"openai/gpt-4o"``.
    """

    name: str
    description: str
    model: str


class AgentDetail(AgentSummary):
    """Full configuration for a single agent.

    Attributes:
        system_prompt: The agent's system-level instruction.
        tools: Names of tools available to the agent.
        max_iterations: Maximum ReAct loop iterations.
    """

    system_prompt: str
    tools: list[str]
    max_iterations: int


class AgentRunRequest(BaseModel):
    """Request body for single-agent execution.

    Attributes:
        message: The user message to send to the agent.
        variables: Optional state variables to seed the execution.
        session_id: Optional conversation id; requests sharing it continue
            one conversation. Omitted, the run remembers nothing.
    """

    message: str = Field(..., min_length=1, description="User message to send.")
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional state variables.",
    )
    session_id: str | None = Field(
        default=None,
        max_length=MAX_SESSION_ID_LENGTH,
        pattern=SESSION_ID_PATTERN,
        description="Conversation to continue; omit for a one-off run.",
    )


class AgentRunResponse(BaseModel):
    """Response from agent execution.

    Attributes:
        agent_name: Name of the agent that executed.
        response: The agent's final text response.
        step_count: Number of AI turns and tool results the run recorded.
        status: Final workflow status string.
        session_id: The conversation the run belongs to, if any.
    """

    agent_name: str
    response: str
    step_count: int
    status: str
    session_id: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get(
    "",
    response_model=list[AgentSummary],
    summary="List all agents",
    description="Returns a summary of every agent defined in the configuration.",
)
def list_agents() -> list[AgentSummary]:
    """List all configured agents.

    Returns:
        A list of agent summaries.

    Raises:
        HTTPException: 503 if the configuration cannot be loaded.
    """
    try:
        profiles = get_agent_profiles()
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return [
        AgentSummary(
            name=profile.name,
            description=profile.description or "",
            model=profile.model.id,
        )
        for profile in profiles
    ]


@router.get(
    "/{agent_name}",
    response_model=AgentDetail,
    summary="Get agent details",
    description="Returns the full configuration for a specific agent.",
)
def get_agent(agent_name: str) -> AgentDetail:
    """Retrieve the full profile for a named agent.

    Args:
        agent_name: The agent identifier from the URL path.

    Returns:
        Full agent configuration detail.

    Raises:
        HTTPException: 404 if the agent name is not found, 503 if the
            configuration cannot be loaded.
    """
    try:
        profiles = get_agent_profiles()
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    matching = [profile for profile in profiles if profile.name == agent_name]
    if not matching:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found.",
        )

    profile = matching[0]
    return AgentDetail(
        name=profile.name,
        description=profile.description or "",
        model=profile.model.id,
        system_prompt=profile.system_prompt,
        tools=list(profile.tools),
        max_iterations=profile.max_iterations,
    )


@router.post(
    "/{agent_name}/run",
    response_model=AgentRunResponse,
    summary="Run an agent",
    description="Run an agent with a user message and return the response.",
)
async def run_agent(
    agent_name: str,
    request: AgentRunRequest,
    http_request: Request,
) -> AgentRunResponse:
    """Run a named agent with a user message.

    With a ``session_id`` the run starts from that conversation's stored
    history, and the messages it adds are saved for the next request.

    Args:
        agent_name: The agent identifier from the URL path.
        request: Execution request with the user message and variables.
        http_request: The incoming request, used to reach the app's open
            toolsets and session store.

    Returns:
        The agent's response with execution metadata.

    Raises:
        HTTPException: 404 if the agent is not found or its configuration
            is invalid, 500 on execution failure, 503 if the session store
            fails.
    """
    agent, session, state = await _prepare_run(agent_name, request, http_request)
    try:
        final_state = await agent.run(state)
    except AgentExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        await session.finish(final_state)
    except SessionStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _run_response(agent_name, final_state, request.session_id)


@router.post(
    "/{agent_name}/run/stream",
    summary="Run an agent, streaming its reply",
    description=(
        "Same as /run, but streams Server-Sent Events: text, tool_call, tool_result, "
        "then completed (same fields as /run) or error."
    ),
    response_class=StreamingResponse,
)
async def stream_agent(
    agent_name: str,
    request: AgentRunRequest,
    http_request: Request,
) -> StreamingResponse:
    """Run a named agent and stream its progress as Server-Sent Events.

    Args:
        agent_name: The agent identifier from the URL path.
        request: Execution request with the user message and variables.
        http_request: The incoming request, used to reach the app's open
            toolsets and session store.

    Returns:
        A ``text/event-stream`` response.

    Raises:
        HTTPException: 404 if the agent is not found or its configuration
            is invalid, and 503 if the session cannot be loaded, both
            before streaming starts. Later failures
            arrive as an ``error`` event.
    """
    agent, session, state = await _prepare_run(agent_name, request, http_request)

    async def events() -> AsyncIterator[str]:
        try:
            async for event in agent.stream(state):
                if not isinstance(event, RunCompleted):
                    yield format_sse(*progress_event(event))
                    continue
                await session.finish(event.state)
                response = _run_response(agent_name, event.state, request.session_id)
                yield format_sse(StreamEventName.COMPLETED, response)
        except (AgentExecutionError, SessionStoreError) as exc:
            yield format_sse(StreamEventName.ERROR, ErrorPayload(detail=str(exc)))

    return sse_response(events())


async def _prepare_run(
    agent_name: str, request: AgentRunRequest, http_request: Request
) -> tuple[BaseAgent, RunSession, State]:
    """Build the agent and its starting state, mapping failures to HTTP errors."""
    try:
        agent = get_agent_by_name(agent_name, toolsets=http_request.app.state.toolsets)
    except ConfigurationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session = RunSession(http_request.app.state.session_store, request.session_id)
    try:
        state = await session.start(agent.profile.system_prompt, request.message, request.variables)
    except SessionStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return agent, session, state


def _run_response(agent_name: str, final_state: State, session_id: str | None) -> AgentRunResponse:
    return AgentRunResponse(
        agent_name=agent_name,
        response=final_response(final_state),
        step_count=final_state.step_count,
        status=final_state.status.value,
        session_id=session_id,
    )
