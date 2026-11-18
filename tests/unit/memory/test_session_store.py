"""Unit tests for session stores: the contract every store meets, plus the in-memory store.

SQLite-specific behaviour is tested in ``test_sqlite_session_store.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from miniagent.core.exceptions import SessionStoreError
from miniagent.core.message import Message, MessageRole, ToolCall
from miniagent.memory.session_store import InMemorySessionStore, SessionStore
from miniagent.memory.sqlite_session_store import SQLiteSessionStore


def _user(text: str) -> Message:
    return Message(role=MessageRole.USER, content=text)


def _ai_calling_tool() -> Message:
    call = ToolCall(id="c1", tool_name="echo", arguments={"text": "x"})
    return Message(
        role=MessageRole.AI,
        content="[tool call]",
        tool_calls=[call],
        model="openai/gpt-4o",
    )


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[SessionStore]:
    if request.param == "memory":
        yield InMemorySessionStore()
    else:
        yield SQLiteSessionStore(tmp_path / "sessions.db")


class TestSessionStoreContract:
    """Every SessionStore implementation behaves the same way."""

    async def test_should_return_empty_history_for_unknown_session(
        self, store: SessionStore
    ) -> None:
        assert await store.load("never-used") == []

    async def test_should_return_appended_messages_in_order(self, store: SessionStore) -> None:
        await store.append("s1", [_user("one"), _user("two")])
        await store.append("s1", [_user("three")])
        assert [message.content for message in await store.load("s1")] == ["one", "two", "three"]

    async def test_should_round_trip_every_message_field(self, store: SessionStore) -> None:
        original = _ai_calling_tool()
        await store.append("s1", [original])
        [loaded] = await store.load("s1")
        assert loaded == original

    async def test_should_keep_sessions_separate(self, store: SessionStore) -> None:
        await store.append("alice", [_user("from alice")])
        await store.append("bob", [_user("from bob")])
        assert [message.content for message in await store.load("alice")] == ["from alice"]

    async def test_should_forget_a_deleted_session(self, store: SessionStore) -> None:
        await store.append("s1", [_user("gone")])
        await store.delete("s1")
        assert await store.load("s1") == []

    async def test_should_ignore_deleting_unknown_session(self, store: SessionStore) -> None:
        await store.delete("never-used")

    async def test_should_accept_appending_nothing(self, store: SessionStore) -> None:
        await store.append("s1", [])
        assert await store.load("s1") == []

    async def test_should_not_expose_internal_storage(self, store: SessionStore) -> None:
        await store.append("s1", [_user("one")])
        loaded = await store.load("s1")
        loaded.append(_user("injected"))
        assert len(await store.load("s1")) == 1

    async def test_should_keep_each_append_contiguous_under_concurrency(
        self, store: SessionStore
    ) -> None:
        batches = [[_user(f"{batch}-{index}") for index in range(5)] for batch in "abcd"]
        await asyncio.gather(*(store.append("shared", batch) for batch in batches))
        contents = [message.content for message in await store.load("shared")]
        assert len(contents) == 20
        for start in range(0, 20, 5):
            assert len({content.split("-")[0] for content in contents[start : start + 5]}) == 1

    @pytest.mark.parametrize(
        "bad_id", ["", "has space", "slash/inside", "trailing-newline\n", "x" * 129]
    )
    async def test_should_reject_invalid_session_ids(
        self, store: SessionStore, bad_id: str
    ) -> None:
        with pytest.raises(SessionStoreError, match="Invalid session id"):
            await store.load(bad_id)


class TestInMemorySessionStore:
    async def test_should_evict_least_recently_used_session_when_full(self) -> None:
        store = InMemorySessionStore(max_sessions=2)
        await store.append("old", [_user("a")])
        await store.append("recent", [_user("b")])
        await store.load("old")  # touching "old" makes "recent" the eviction candidate
        await store.append("new", [_user("c")])
        assert await store.load("recent") == []
        assert await store.load("old") != []

    def test_should_reject_non_positive_capacity(self) -> None:
        with pytest.raises(SessionStoreError, match="max_sessions must be at least 1"):
            InMemorySessionStore(max_sessions=0)
