"""Tests for POST /agents/{name}/run/stream (Server-Sent Events)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from miniagent.agent.agent import Agent
from miniagent.api.app import create_app
from miniagent.core.exceptions import ConfigurationError, SessionStoreError
from miniagent.core.message import ToolCall
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.model_config import ModelConfig
from miniagent.tool.function_tool import tool
from tests.unit.llm.fakes import ScriptedProvider, ScriptedTurn, text_turn, tool_turn

_GET_AGENT = "miniagent.api.routes.agents.get_agent_by_name"
_PROFILE = AgentProfile(
    name="bot", model=ModelConfig(id="openai/gpt-4o"), system_prompt="Be brief.", tools=["add"]
)


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


@pytest.fixture()
def app() -> FastAPI:
    return create_app()


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@contextmanager
def _serve(*turns: ScriptedTurn) -> Iterator[ScriptedProvider]:
    """Serve agent 'bot' backed by a provider that plays back ``turns``."""
    provider = ScriptedProvider(*turns)
    with patch(_GET_AGENT, return_value=Agent(_PROFILE, provider, tools={"add": add})):
        yield provider


def _events(client: TestClient, body: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """POST to the stream endpoint and parse every event."""
    response = client.post("/agents/bot/run/stream", json=body)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    parsed = []
    for block in response.text.strip().split("\n\n"):
        event_line, data_line = block.split("\n")
        parsed.append(
            (event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: ")))
        )
    return parsed


class TestStreamAgent:
    def test_should_stream_text_then_completed(self, client: TestClient) -> None:
        with _serve(text_turn("Hel", "lo!")):
            events = _events(client, {"message": "hi"})

        assert events[:2] == [("text", {"delta": "Hel"}), ("text", {"delta": "lo!"})]
        name, completed = events[-1]
        assert name == "completed"
        assert completed == {
            "agent_name": "bot",
            "response": "Hello!",
            "step_count": 1,
            "status": "completed",
            "session_id": None,
        }

    def test_should_stream_tool_calls_and_results(self, client: TestClient) -> None:
        call = ToolCall(id="c1", tool_name="add", arguments={"a": 2, "b": 3})
        with _serve(tool_turn(call), text_turn("5")):
            events = _events(client, {"message": "2+3?"})

        assert [name for name, _ in events] == ["tool_call", "tool_result", "text", "completed"]
        assert events[0][1] == {"id": "c1", "name": "add", "arguments": {"a": 2, "b": 3}}
        assert events[1][1] == {"id": "c1", "name": "add", "content": "5", "is_error": False}

    def test_should_continue_and_save_the_session(self, app: FastAPI, client: TestClient) -> None:
        with _serve(text_turn("Hi Ada."), text_turn("You are Ada.")) as provider:
            _events(client, {"message": "I am Ada.", "session_id": "s1"})
            _events(client, {"message": "Who am I?", "session_id": "s1"})

        assert [m.content for m in provider.received[1]] == [
            "Be brief.",
            "I am Ada.",
            "Hi Ada.",
            "Who am I?",
        ]
        stored = asyncio.run(app.state.session_store.load("s1"))
        assert len(stored) == 4

    def test_should_report_run_failures_as_an_error_event(self, client: TestClient) -> None:
        with _serve():  # no scripted turns: the provider fails on the first call
            events = _events(client, {"message": "hi"})

        [(name, payload)] = events
        assert name == "error"
        assert "AgentExecutionError" in payload["detail"]

    def test_should_return_404_before_streaming_for_unknown_agents(
        self, client: TestClient
    ) -> None:
        with patch(_GET_AGENT, side_effect=ConfigurationError("Agent 'bot' not found")):
            response = client.post("/agents/bot/run/stream", json={"message": "hi"})
        assert response.status_code == 404

    def test_should_return_503_before_streaming_when_history_cannot_load(
        self, app: FastAPI, client: TestClient
    ) -> None:
        failing_store = MagicMock()
        failing_store.load = AsyncMock(side_effect=SessionStoreError("disk full"))
        app.state.session_store = failing_store

        with _serve(text_turn("unused")):
            response = client.post(
                "/agents/bot/run/stream", json={"message": "hi", "session_id": "s1"}
            )
        assert response.status_code == 503
