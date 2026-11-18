"""Session stores: conversation history that outlives a single run.

An agent's ``run()`` starts from whatever ``State`` it is given. A
``SessionStore`` keeps each conversation's messages under a session id, so
the next request can continue where the last one stopped::

    history = await store.load(session_id)
    final_state = await agent.run(State(messages=[*history, user_message]))
    await store.append(session_id, final_state.messages[len(history):])

Two implementations ship: ``InMemorySessionStore`` (per process, bounded)
and ``SQLiteSessionStore`` (durable, in ``miniagent.memory.sqlite_session_store``).
Stores hold the full history; pair them with a ``ContextPolicy`` so long
conversations still fit the model's context.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence

from miniagent.core.exceptions import SessionStoreError
from miniagent.core.message import Message

MAX_SESSION_ID_LENGTH = 128
SESSION_ID_PATTERN = r"^[A-Za-z0-9_.:-]+$"
DEFAULT_MAX_SESSIONS = 1000

_SESSION_ID = re.compile(SESSION_ID_PATTERN)


class SessionStore(ABC):
    """Loads and appends the messages of conversations, keyed by session id.

    Session ids are 1-128 characters from ``A-Z a-z 0-9 _ . : -``. Each
    ``append`` lands as one contiguous block, so a tool request and its
    results are never interleaved with another request's messages.
    """

    @abstractmethod
    async def load(self, session_id: str) -> list[Message]:
        """Return the session's messages, oldest first; empty if unknown.

        Raises:
            SessionStoreError: If the id is invalid or storage fails.
        """

    @abstractmethod
    async def append(self, session_id: str, messages: Sequence[Message]) -> None:
        """Add ``messages`` to the end of the session, creating it if needed.

        Raises:
            SessionStoreError: If the id is invalid or storage fails.
        """

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Forget the session; deleting an unknown session does nothing.

        Raises:
            SessionStoreError: If the id is invalid or storage fails.
        """


def validate_session_id(session_id: str) -> str:
    """Return ``session_id`` if it follows the session id rule.

    Raises:
        SessionStoreError: If it is empty, too long, or has other characters.
    """
    if len(session_id) > MAX_SESSION_ID_LENGTH or not _SESSION_ID.fullmatch(session_id):
        raise SessionStoreError(
            f"Invalid session id {session_id!r}: use 1-{MAX_SESSION_ID_LENGTH} "
            f"characters from A-Z, a-z, 0-9, '_', '.', ':' and '-'."
        )
    return session_id


class InMemorySessionStore(SessionStore):
    """Keeps sessions in process memory; lost on restart.

    Holds at most ``max_sessions`` conversations and evicts the least
    recently used one beyond that, so memory stays bounded. Safe for
    concurrent use within one event loop.
    """

    def __init__(self, max_sessions: int = DEFAULT_MAX_SESSIONS) -> None:
        """Create an empty store.

        Args:
            max_sessions: Most conversations kept before evicting the least
                recently used.

        Raises:
            SessionStoreError: If ``max_sessions`` is less than 1.
        """
        if max_sessions < 1:
            raise SessionStoreError(f"max_sessions must be at least 1, got {max_sessions}.")
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, list[Message]] = OrderedDict()

    async def load(self, session_id: str) -> list[Message]:
        """Return a copy of the session's messages; empty if unknown."""
        messages = self._sessions.get(validate_session_id(session_id))
        if messages is None:
            return []
        self._sessions.move_to_end(session_id)
        return list(messages)

    async def append(self, session_id: str, messages: Sequence[Message]) -> None:
        """Add ``messages``, evicting the least recently used session if full."""
        validate_session_id(session_id)
        if not messages:
            return
        self._sessions.setdefault(session_id, []).extend(messages)
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)

    async def delete(self, session_id: str) -> None:
        """Forget the session."""
        self._sessions.pop(validate_session_id(session_id), None)
