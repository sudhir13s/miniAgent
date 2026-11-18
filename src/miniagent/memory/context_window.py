"""Context window policies: choose which messages are sent to the model.

An agent's history grows with every turn and every tool result, and a long
run eventually exceeds the model's context limit. A ``ContextPolicy`` picks
the messages for each model call; the agent's ``State`` always keeps the full
history, so trimming never loses data.

``TokenBudgetPolicy`` keeps the newest messages that fit a token budget and
always keeps:

* the leading system prompt,
* the latest user message (the task being worked on), and
* the newest turn, even if it alone is over budget.

An AI message that requested tools and the tool results answering it form
one unit that is kept or dropped as a whole: providers reject a tool result
whose request is missing, and a request whose results are missing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from miniagent.core.exceptions import ConfigurationError
from miniagent.core.message import Message, MessageRole

logger = logging.getLogger(__name__)

TokenCounter = Callable[[Sequence[Message]], int]
"""Counts the tokens a sequence of messages costs for the agent's model."""


class ContextPolicy(ABC):
    """Decides which part of the history is sent to the model on each call."""

    @abstractmethod
    def fit(self, messages: Sequence[Message], count_tokens: TokenCounter) -> list[Message]:
        """Return the messages to send, in their original order.

        Args:
            messages: The full history, starting with any system prompt.
            count_tokens: Counts tokens for the agent's model.

        Returns:
            A new list; ``messages`` is never modified.
        """


class TokenBudgetPolicy(ContextPolicy):
    """Keep the newest messages that fit ``max_tokens``, oldest dropped first.

    The budget covers messages only; tool schemas and the model's reply need
    room of their own within the model's context limit.

    Attributes:
        max_tokens: Most tokens the sent messages may cost.
    """

    def __init__(self, max_tokens: int) -> None:
        """Create a policy with the given budget.

        Args:
            max_tokens: Most tokens the sent messages may cost.

        Raises:
            ConfigurationError: If ``max_tokens`` is less than 1.
        """
        if max_tokens < 1:
            raise ConfigurationError(f"max_tokens must be at least 1, got {max_tokens}.")
        self.max_tokens = max_tokens

    def fit(self, messages: Sequence[Message], count_tokens: TokenCounter) -> list[Message]:
        """Return the newest messages that fit, plus the pinned ones.

        Args:
            messages: The full history, starting with any system prompt.
            count_tokens: Counts tokens for the agent's model.

        Returns:
            The messages to send, in their original order.
        """
        history = list(messages)
        if count_tokens(history) <= self.max_tokens:
            return history

        system_count = _leading_system_count(history)
        system_prompt = history[:system_count]
        units = _group_into_units(history[system_count:])

        remaining = self.max_tokens - count_tokens(system_prompt)
        task = _mark_task(units)
        if task is not None:
            remaining -= count_tokens(task.messages)
        kept = _newest_units_within(units, remaining, count_tokens)

        window = system_prompt + [
            message
            for index, unit in enumerate(units)
            if unit.is_task or index in kept
            for message in unit.messages
        ]
        logger.debug("Context window kept %d of %d messages", len(window), len(history))
        if count_tokens(window) > self.max_tokens:
            logger.warning(
                "Pinned messages exceed the context budget of %d tokens; sending them anyway.",
                self.max_tokens,
            )
        return window


def _leading_system_count(history: Sequence[Message]) -> int:
    """Return how many messages at the start of ``history`` are system messages."""
    count = 0
    while count < len(history) and history[count].role == MessageRole.SYSTEM:
        count += 1
    return count


@dataclass(slots=True)
class _Unit:
    """Messages that are kept or dropped together.

    Attributes:
        messages: One message, or an AI tool request followed by its results.
        is_task: Whether this unit holds the latest user message.
    """

    messages: list[Message]
    is_task: bool = False

    @property
    def is_orphan(self) -> bool:
        """A unit that starts with a tool result has lost its request."""
        return self.messages[0].role == MessageRole.TOOL


def _group_into_units(messages: Sequence[Message]) -> list[_Unit]:
    """Group messages into units that are kept or dropped together.

    A tool result joins the unit of the AI message before it, so a tool
    request and its results stay together. A tool result with no AI message
    before it starts an orphan unit of its own.
    """
    units: list[_Unit] = []
    for message in messages:
        if (
            message.role == MessageRole.TOOL
            and units
            and units[-1].messages[0].role == MessageRole.AI
        ):
            units[-1].messages.append(message)
        else:
            units.append(_Unit([message]))
    return units


def _mark_task(units: Sequence[_Unit]) -> _Unit | None:
    """Flag the newest unit holding a user message as the task and return it."""
    for unit in reversed(units):
        if unit.messages[0].role == MessageRole.USER:
            unit.is_task = True
            return unit
    return None


def _newest_units_within(
    units: Sequence[_Unit], budget: int, count_tokens: TokenCounter
) -> set[int]:
    """Pick unit indices from newest to oldest until the next one does not fit.

    The task unit is left out (the caller always keeps it and has already
    taken its cost from ``budget``) and so are orphan units (a tool result
    whose request is gone); otherwise the newest unit is always picked.
    Stopping at the first unit that does not fit keeps the window a
    contiguous stretch of recent history.
    """
    newest_index = len(units) - 1
    kept: set[int] = set()
    for index in range(newest_index, -1, -1):
        unit = units[index]
        if unit.is_task or unit.is_orphan:
            continue
        cost = count_tokens(unit.messages)
        if cost > budget and index != newest_index:
            break
        kept.add(index)
        budget -= cost
    return kept
