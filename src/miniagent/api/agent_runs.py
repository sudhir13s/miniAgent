"""What the agent run endpoints share: session handling and event payloads.

``POST /agents/{name}/run`` and ``POST /agents/{name}/run/stream`` start and
finish a run the same way; only the delivery differs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from miniagent.agent.events import AgentEvent, RunCompleted, TextDelta, ToolCallStarted
from miniagent.api.stream_events import (
    StreamEventName,
    TextPayload,
    ToolCallPayload,
    ToolResultPayload,
)
from miniagent.core.message import Message, MessageRole
from miniagent.core.state import State
from miniagent.memory.session_store import SessionStore

NO_RESPONSE = "No response generated."


class RunSession:
    """One request's conversation: loads its history and saves what the run adds.

    Without a session id nothing is loaded or saved. Store failures raise
    ``SessionStoreError`` for the endpoint to report.
    """

    def __init__(self, store: SessionStore, session_id: str | None) -> None:
        """Create the session for one request.

        Args:
            store: The application's session store.
            session_id: The conversation to continue, or ``None``.
        """
        self._store = store
        self._session_id = session_id
        self._prefix_length = 0

    async def start(
        self, system_prompt: str, user_message: str, variables: dict[str, Any]
    ) -> State:
        """Build the run's starting state: system prompt, history, new message."""
        history = await self._store.load(self._session_id) if self._session_id else []
        prefix = [Message(role=MessageRole.SYSTEM, content=system_prompt), *history]
        self._prefix_length = len(prefix)
        new_message = Message(role=MessageRole.USER, content=user_message)
        return State(variables=variables, messages=[*prefix, new_message])

    async def finish(self, final_state: State) -> None:
        """Save the messages the run added (never the system prompt)."""
        if self._session_id:
            await self._store.append(self._session_id, final_state.messages[self._prefix_length :])


def final_response(final_state: State) -> str:
    """Return the last AI answer's text, or a placeholder if there is none.

    AI turns that only requested tools are skipped: a run that stops at
    ``max_iterations`` can end on one, and it holds no answer.
    """
    answers = [
        message.content
        for message in final_state.messages
        if message.role == MessageRole.AI and not message.tool_calls
    ]
    return answers[-1] if answers else NO_RESPONSE


def progress_event(event: AgentEvent) -> tuple[StreamEventName, BaseModel]:
    """Name and payload for a streamed event other than ``RunCompleted``."""
    if isinstance(event, RunCompleted):
        raise ValueError("RunCompleted is reported by the endpoint, with its response.")
    if isinstance(event, TextDelta):
        return StreamEventName.TEXT, TextPayload(delta=event.text)
    if isinstance(event, ToolCallStarted):
        call = event.tool_call
        return StreamEventName.TOOL_CALL, ToolCallPayload(
            id=call.id, name=call.tool_name, arguments=call.arguments
        )
    result = event.result
    return StreamEventName.TOOL_RESULT, ToolResultPayload(
        id=result.tool_call_id,
        name=result.tool_name,
        content=result.content,
        is_error=result.is_error,
    )
