"""Unit tests for behaviour specific to SQLiteSessionStore.

The contract shared with every store (ordering, isolation, concurrent
appends, id validation) runs against SQLite in ``test_session_store.py``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from miniagent.core.exceptions import SessionStoreError
from miniagent.core.message import Message, MessageRole
from miniagent.memory.sqlite_session_store import SQLiteSessionStore


def _user(text: str) -> Message:
    return Message(role=MessageRole.USER, content=text)


class TestSQLiteSessionStore:
    async def test_should_persist_across_store_instances(self, tmp_path: Path) -> None:
        database = tmp_path / "sessions.db"
        await SQLiteSessionStore(database).append("s1", [_user("remembered")])
        reopened = SQLiteSessionStore(database)
        assert [message.content for message in await reopened.load("s1")] == ["remembered"]

    async def test_should_continue_positions_after_reopening(self, tmp_path: Path) -> None:
        database = tmp_path / "sessions.db"
        await SQLiteSessionStore(database).append("s1", [_user("one"), _user("two")])
        await SQLiteSessionStore(database).append("s1", [_user("three")])
        contents = [message.content for message in await SQLiteSessionStore(database).load("s1")]
        assert contents == ["one", "two", "three"]

    async def test_should_wrap_storage_failures(self, tmp_path: Path) -> None:
        database = tmp_path / "sessions.db"
        store = SQLiteSessionStore(database)
        database.write_bytes(b"this is not a sqlite database" * 100)
        with pytest.raises(SessionStoreError, match="Could not read session 's1'") as excinfo:
            await store.load("s1")
        assert isinstance(excinfo.value.cause, sqlite3.DatabaseError)

    def test_should_fail_fast_when_database_cannot_be_created(self, tmp_path: Path) -> None:
        with pytest.raises(SessionStoreError, match="Could not open session database"):
            SQLiteSessionStore(tmp_path / "missing-dir" / "sessions.db")
