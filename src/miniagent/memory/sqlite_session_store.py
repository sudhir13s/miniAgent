"""SQLite-backed session store: conversation history that survives restarts.

Uses only the standard library. Each load, append and delete opens a
short-lived connection in a worker thread, so the event loop never blocks on disk I/O
and the store is safe to share across concurrent requests. Appends take a
write lock before reading the next position, so concurrent appends to one
session (even from several processes) each land as a contiguous block.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path

from miniagent.core.exceptions import SessionStoreError
from miniagent.core.message import Message
from miniagent.memory.session_store import SessionStore, validate_session_id

# Seconds a connection waits for another writer's lock before failing.
_LOCK_TIMEOUT_SECONDS = 30.0

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS session_messages (
    session_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    message_json TEXT NOT NULL,
    PRIMARY KEY (session_id, position)
)
"""
_SELECT_MESSAGES = (
    "SELECT message_json FROM session_messages WHERE session_id = ? ORDER BY position"
)
_SELECT_NEXT_POSITION = (
    "SELECT COALESCE(MAX(position) + 1, 0) FROM session_messages WHERE session_id = ?"
)
_INSERT_MESSAGE = (
    "INSERT INTO session_messages (session_id, position, message_json) VALUES (?, ?, ?)"
)
_DELETE_SESSION = "DELETE FROM session_messages WHERE session_id = ?"


class SQLiteSessionStore(SessionStore):
    """Keeps sessions in a SQLite database file.

    Attributes:
        path: The database file; created with its table by the constructor.
    """

    def __init__(self, path: str | Path) -> None:
        """Open (or create) the database and its table.

        Args:
            path: Database file path. The parent directory must exist.

        Raises:
            SessionStoreError: If the database cannot be created or opened.
        """
        self.path = Path(path)
        try:
            with self._transaction() as connection:
                connection.execute(_CREATE_TABLE)
        except sqlite3.Error as exc:
            raise SessionStoreError(
                f"Could not open session database at {self.path}.", cause=exc
            ) from exc

    async def load(self, session_id: str) -> list[Message]:
        """Return the session's messages, oldest first; empty if unknown."""
        validate_session_id(session_id)
        try:
            rows = await asyncio.to_thread(self._select_messages, session_id)
        except sqlite3.Error as exc:
            raise SessionStoreError(f"Could not read session '{session_id}'.", cause=exc) from exc
        return [Message.model_validate_json(message_json) for message_json in rows]

    async def append(self, session_id: str, messages: Sequence[Message]) -> None:
        """Add ``messages`` to the end of the session in one transaction."""
        validate_session_id(session_id)
        if not messages:
            return
        payloads = [message.model_dump_json() for message in messages]
        try:
            await asyncio.to_thread(self._insert_messages, session_id, payloads)
        except sqlite3.Error as exc:
            raise SessionStoreError(f"Could not save session '{session_id}'.", cause=exc) from exc

    async def delete(self, session_id: str) -> None:
        """Forget the session."""
        validate_session_id(session_id)
        try:
            await asyncio.to_thread(self._delete_session, session_id)
        except sqlite3.Error as exc:
            raise SessionStoreError(f"Could not delete session '{session_id}'.", cause=exc) from exc

    def _select_messages(self, session_id: str) -> list[str]:
        with self._transaction() as connection:
            rows = connection.execute(_SELECT_MESSAGES, (session_id,)).fetchall()
        return [row[0] for row in rows]

    def _insert_messages(self, session_id: str, payloads: Sequence[str]) -> None:
        with self._transaction(write=True) as connection:
            (next_position,) = connection.execute(_SELECT_NEXT_POSITION, (session_id,)).fetchone()
            connection.executemany(
                _INSERT_MESSAGE,
                [
                    (session_id, next_position + offset, payload)
                    for offset, payload in enumerate(payloads)
                ],
            )

    def _delete_session(self, session_id: str) -> None:
        with self._transaction(write=True) as connection:
            connection.execute(_DELETE_SESSION, (session_id,))

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside a transaction, committed on success.

        ``write=True`` takes the database write lock up front (``BEGIN
        IMMEDIATE``), so a read-then-insert cannot race another writer.
        """
        with closing(
            sqlite3.connect(self.path, timeout=_LOCK_TIMEOUT_SECONDS, isolation_level=None)
        ) as connection:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
