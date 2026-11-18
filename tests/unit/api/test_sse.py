"""Unit tests for the Server-Sent Events helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from miniagent.api.sse import format_sse, sse_response
from miniagent.api.stream_events import StreamEventName, TextPayload


class _Invoice(BaseModel):
    number: str
    total: float


class _Completed(BaseModel):
    variables: dict[str, Any]
    at: datetime


class TestFormatSse:
    def test_should_write_event_and_data_lines_ending_with_a_blank_line(self) -> None:
        event = format_sse(StreamEventName.TEXT, TextPayload(delta="Hi"))
        assert event == 'event: text\ndata: {"delta":"Hi"}\n\n'

    def test_should_serialise_nested_models_and_datetimes(self) -> None:
        payload = _Completed(
            variables={"output": _Invoice(number="A-1", total=9.5)},
            at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        data_line = format_sse(StreamEventName.COMPLETED, payload).split("\n")[1]
        assert json.loads(data_line.removeprefix("data: ")) == {
            "variables": {"output": {"number": "A-1", "total": 9.5}},
            "at": "2026-01-01T00:00:00Z",
        }

    def test_should_keep_multiline_text_on_one_data_line(self) -> None:
        event = format_sse(StreamEventName.TEXT, TextPayload(delta="line one\nline two"))
        assert event.count("\n") == 3  # event line, data line, blank line


class TestSseResponse:
    def test_should_be_an_unbuffered_event_stream(self) -> None:
        async def events() -> Any:
            yield ""

        response = sse_response(events())

        assert response.media_type == "text/event-stream"
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"
