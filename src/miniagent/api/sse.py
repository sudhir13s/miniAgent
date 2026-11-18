"""Server-Sent Events helpers shared by the streaming endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from miniagent.api.stream_events import StreamEventName

# Stop browsers and proxies (e.g. nginx) from buffering the stream.
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_SSE_MEDIA_TYPE = "text/event-stream"


def format_sse(name: StreamEventName, payload: BaseModel) -> str:
    """Format one Server-Sent Event.

    Args:
        name: The event name.
        payload: The event's data, written as one line of JSON.

    Returns:
        The ``event:`` and ``data:`` lines, terminated by a blank line.
    """
    return f"event: {name.value}\ndata: {payload.model_dump_json()}\n\n"


def sse_response(events: AsyncIterator[str]) -> StreamingResponse:
    """Wrap formatted events in a ``text/event-stream`` response."""
    return StreamingResponse(events, media_type=_SSE_MEDIA_TYPE, headers=_SSE_HEADERS)
