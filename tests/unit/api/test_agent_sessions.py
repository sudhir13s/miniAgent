"""Tests for conversation sessions on POST /agents/{name}/run."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from miniagent.api.app import create_app
from miniagent.core.exceptions import AgentExecutionError, SessionStoreError
from miniagent.core.message import Message, MessageRole
from miniagent.core.state import State
from miniagent.memory.session_store import SessionStore

_GET_AGENT = "miniagent.api.routes.agents.get_agent_by_name"


class _ReplyingAgent:
    """Stands in for an agent: replies to the latest user message and records what it saw."""

    def __init__(self) -> None:
        self.profile = MagicMock(system_prompt="Be helpful.")
        self.received: list[list[str]] = []

    async def run(self, state: State) -> State:
        self.received.append([message.content for message in state.messages])
        latest = [m for m in state.messages if m.role == MessageRole.USER][-1]
        reply = Message(role=MessageRole.AI, content=f"reply to {latest.content}")
        return state.add_message(reply).increment_step()


@pytest.fixture()
def app() -> FastAPI:
    return create_app()


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def agent() -> Iterator[_ReplyingAgent]:
    replying_agent = _ReplyingAgent()
    with patch(_GET_AGENT, return_value=replying_agent):
        yield replying_agent


def _run(client: TestClient, message: str, session_id: str | None = None) -> dict[str, object]:
    body: dict[str, object] = {"message": message}
    if session_id is not None:
        body["session_id"] = session_id
    response = client.post("/agents/bot/run", json=body)
    assert response.status_code == 200, response.text
    payload: dict[str, object] = response.json()
    return payload


class TestRunWithSession:
    def test_should_continue_the_conversation_with_the_same_session_id(
        self, client: TestClient, agent: _ReplyingAgent
    ) -> None:
        _run(client, "my name is Ada", session_id="chat-1")
        _run(client, "what is my name?", session_id="chat-1")

        assert agent.received[1] == [
            "Be helpful.",
            "my name is Ada",
            "reply to my name is Ada",
            "what is my name?",
        ]

    def test_should_echo_the_session_id(self, client: TestClient, agent: _ReplyingAgent) -> None:
        assert _run(client, "hi", session_id="chat-1")["session_id"] == "chat-1"

    def test_should_store_new_messages_but_not_the_system_prompt(
        self, app: FastAPI, client: TestClient, agent: _ReplyingAgent
    ) -> None:
        _run(client, "first", session_id="chat-1")
        _run(client, "second", session_id="chat-1")

        contents = [message.content for message in _load(app.state.session_store, "chat-1")]
        assert contents == ["first", "reply to first", "second", "reply to second"]

    def test_should_keep_sessions_apart(self, client: TestClient, agent: _ReplyingAgent) -> None:
        _run(client, "secret for one", session_id="one")
        _run(client, "hello", session_id="two")
        assert "secret for one" not in agent.received[1]

    def test_should_not_remember_anything_without_a_session_id(
        self, app: FastAPI, client: TestClient, agent: _ReplyingAgent
    ) -> None:
        first = _run(client, "remember me")
        _run(client, "do you?")

        assert first["session_id"] is None
        assert agent.received[1] == ["Be helpful.", "do you?"]

    @pytest.mark.parametrize("bad_id", ["", "has space", "x" * 129])
    def test_should_reject_invalid_session_ids(
        self, client: TestClient, agent: _ReplyingAgent, bad_id: str
    ) -> None:
        response = client.post("/agents/bot/run", json={"message": "hi", "session_id": bad_id})
        assert response.status_code == 422

    def test_should_return_503_when_the_store_fails(
        self, app: FastAPI, client: TestClient, agent: _ReplyingAgent
    ) -> None:
        failing_store = MagicMock()
        failing_store.load = AsyncMock(side_effect=SessionStoreError("disk full"))
        app.state.session_store = failing_store

        response = client.post("/agents/bot/run", json={"message": "hi", "session_id": "s1"})

        assert response.status_code == 503
        assert "disk full" in response.json()["detail"]

    def test_should_not_save_when_the_run_fails(self, app: FastAPI, client: TestClient) -> None:
        failing_agent = MagicMock()
        failing_agent.profile.system_prompt = "Be helpful."
        failing_agent.run = AsyncMock(side_effect=AgentExecutionError("LLM down"))

        with patch(_GET_AGENT, return_value=failing_agent):
            response = client.post("/agents/bot/run", json={"message": "hi", "session_id": "s1"})

        assert response.status_code == 500
        assert _load(app.state.session_store, "s1") == []


def _load(store: SessionStore, session_id: str) -> list[Message]:
    """Read a session synchronously, outside the app's event loop."""
    return asyncio.run(store.load(session_id))
