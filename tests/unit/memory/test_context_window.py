"""Unit tests for the context window policies."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from miniagent.core.exceptions import ConfigurationError
from miniagent.core.message import Message, MessageRole, ToolCall
from miniagent.memory.context_window import TokenBudgetPolicy

# ---------------------------------------------------------------------------
# Helpers: every message costs exactly 10 tokens, so budgets are easy to read.
# ---------------------------------------------------------------------------

TOKENS_PER_MESSAGE = 10


def _count_tokens(messages: Sequence[Message]) -> int:
    return TOKENS_PER_MESSAGE * len(messages)


def _system(text: str = "You are helpful.") -> Message:
    return Message(role=MessageRole.SYSTEM, content=text)


def _user(text: str) -> Message:
    return Message(role=MessageRole.USER, content=text)


def _ai(text: str) -> Message:
    return Message(role=MessageRole.AI, content=text)


def _ai_calling(*call_ids: str) -> Message:
    calls = [ToolCall(id=call_id, tool_name="echo") for call_id in call_ids]
    return Message(role=MessageRole.AI, content="[tool call]", tool_calls=calls)


def _tool_result(call_id: str) -> Message:
    return Message(
        role=MessageRole.TOOL,
        content=f"result {call_id}",
        tool_call_id=call_id,
        tool_name="echo",
    )


def _contents(messages: Sequence[Message]) -> list[str]:
    return [message.content for message in messages]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTokenBudgetPolicy:
    def test_should_return_history_unchanged_when_it_fits(self) -> None:
        history = [_system(), _user("hi"), _ai("hello")]
        fitted = TokenBudgetPolicy(max_tokens=30).fit(history, _count_tokens)
        assert fitted == history

    def test_should_drop_oldest_turns_first_when_over_budget(self) -> None:
        history = [_system(), _user("q1"), _ai("a1"), _user("q2"), _ai("a2")]
        fitted = TokenBudgetPolicy(max_tokens=30).fit(history, _count_tokens)
        assert _contents(fitted) == ["You are helpful.", "q2", "a2"]

    def test_should_keep_latest_user_message_when_tool_rounds_fill_the_budget(self) -> None:
        history = [
            _system(),
            _user("the task"),
            _ai_calling("c1"),
            _tool_result("c1"),
            _ai_calling("c2"),
            _tool_result("c2"),
        ]
        fitted = TokenBudgetPolicy(max_tokens=40).fit(history, _count_tokens)
        assert _contents(fitted) == ["You are helpful.", "the task", "[tool call]", "result c2"]

    def test_should_never_separate_a_tool_call_from_its_results(self) -> None:
        history = [
            _system(),
            _user("task"),
            _ai_calling("c1", "c2"),
            _tool_result("c1"),
            _tool_result("c2"),
        ]
        # Room for system + task + two messages: the three-message tool round
        # cannot be cut in half, so it stays whole and the budget is exceeded.
        fitted = TokenBudgetPolicy(max_tokens=40).fit(history, _count_tokens)
        assert _contents(fitted) == [
            "You are helpful.",
            "task",
            "[tool call]",
            "result c1",
            "result c2",
        ]

    def test_should_keep_whole_tool_round_or_none_of_it(self) -> None:
        history = [
            _system(),
            _user("task"),
            _ai_calling("c1"),
            _tool_result("c1"),
            _ai("summary"),
        ]
        fitted = TokenBudgetPolicy(max_tokens=40).fit(history, _count_tokens)
        assert _contents(fitted) == ["You are helpful.", "task", "summary"]

    def test_should_drop_orphaned_tool_results_when_trimming(self) -> None:
        history = [_system(), _tool_result("lost"), _user("q1"), _ai("a1"), _user("q2")]
        fitted = TokenBudgetPolicy(max_tokens=40).fit(history, _count_tokens)
        assert MessageRole.TOOL not in [message.role for message in fitted]
        assert _contents(fitted) == ["You are helpful.", "q1", "a1", "q2"]

    def test_should_keep_newest_turn_even_when_it_alone_exceeds_budget(self) -> None:
        history = [_user("old"), _ai("older answer"), _user("newest")]
        fitted = TokenBudgetPolicy(max_tokens=1).fit(history, _count_tokens)
        assert _contents(fitted) == ["newest"]

    def test_should_keep_messages_in_original_order(self) -> None:
        history = [_system(), _user("q1"), _ai("a1"), _user("q2"), _ai("a2"), _ai("a3")]
        fitted = TokenBudgetPolicy(max_tokens=40).fit(history, _count_tokens)
        assert _contents(fitted) == ["You are helpful.", "q2", "a2", "a3"]

    def test_should_handle_empty_history(self) -> None:
        assert TokenBudgetPolicy(max_tokens=10).fit([], _count_tokens) == []

    def test_should_not_mutate_the_given_history(self) -> None:
        history = [_system(), _user("q1"), _ai("a1"), _user("q2")]
        original = list(history)
        TokenBudgetPolicy(max_tokens=20).fit(history, _count_tokens)
        assert history == original

    @pytest.mark.parametrize("max_tokens", [0, -5])
    def test_should_reject_non_positive_budget(self, max_tokens: int) -> None:
        with pytest.raises(ConfigurationError, match="max_tokens must be at least 1"):
            TokenBudgetPolicy(max_tokens=max_tokens)
