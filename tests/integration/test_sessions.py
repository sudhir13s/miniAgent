"""End-to-end sessions: the REST API, a real Agent, LiteLLM (patched) and SQLite.

No live model is called; ``litellm.acompletion`` is patched, while token
counting uses LiteLLM's real (offline) tokenizer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from miniagent.api.app import create_app
from miniagent.api.dependencies import reset_profile_cache
from miniagent.memory.sqlite_session_store import SQLiteSessionStore
from tests.integration.conftest import make_litellm_response

_ACOMPLETION = "miniagent.llm.litellm_provider.litellm.acompletion"
_LONG_FIRST_MESSAGE = "background " * 300
_CONFIG = """
agents:
  - name: chatbot
    system_prompt: "You are a friendly assistant."
    model: {id: openai/gpt-4o}
  - name: budgeted
    system_prompt: "You are a friendly assistant."
    model: {id: openai/gpt-4o}
    max_context_tokens: 200
"""


@pytest.fixture()
def configured_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the API at a test config and a SQLite session database."""
    config_file = tmp_path / "agents.yaml"
    config_file.write_text(_CONFIG, encoding="utf-8")
    database = tmp_path / "sessions.db"
    monkeypatch.setenv("AGENTS_CONFIG_PATH", str(config_file))
    monkeypatch.setenv("SESSION_DB_PATH", str(database))
    reset_profile_cache()
    yield database
    reset_profile_cache()


def _post(agent_name: str, message: str, completions: AsyncMock) -> dict[str, Any]:
    """Send one message in session 'chat-1' through a freshly created app."""
    with patch(_ACOMPLETION, completions), TestClient(create_app()) as client:
        response = client.post(
            f"/agents/{agent_name}/run", json={"message": message, "session_id": "chat-1"}
        )
    assert response.status_code == 200, response.text
    payload: dict[str, Any] = response.json()
    return payload


def _sent_messages(completions: AsyncMock, call_index: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = completions.call_args_list[call_index].kwargs["messages"]
    return messages


def test_conversation_survives_an_api_restart(configured_api: Path) -> None:
    completions = AsyncMock(
        side_effect=[
            make_litellm_response("Nice to meet you, Ada."),
            make_litellm_response("Your name is Ada."),
        ]
    )

    _post("chatbot", "My name is Ada.", completions)
    second = _post("chatbot", "What is my name?", completions)

    assert second["response"] == "Your name is Ada."
    assert second["status"] == "completed"
    assert _sent_messages(completions, 1) == [
        {"role": "system", "content": "You are a friendly assistant."},
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Nice to meet you, Ada."},
        {"role": "user", "content": "What is my name?"},
    ]


def test_long_session_is_trimmed_for_the_model_but_kept_in_storage(
    configured_api: Path,
) -> None:
    completions = AsyncMock(
        side_effect=[make_litellm_response("Noted."), make_litellm_response("Hello again.")]
    )

    _post("budgeted", _LONG_FIRST_MESSAGE, completions)
    _post("budgeted", "Hi!", completions)

    sent_contents = [message["content"] for message in _sent_messages(completions, 1)]
    assert sent_contents == ["You are a friendly assistant.", "Noted.", "Hi!"]
    stored = asyncio.run(SQLiteSessionStore(configured_api).load("chat-1"))
    assert [message.content for message in stored] == [
        _LONG_FIRST_MESSAGE,
        "Noted.",
        "Hi!",
        "Hello again.",
    ]
