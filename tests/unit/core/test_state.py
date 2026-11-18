"""Tests for State and its immutable transitions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miniagent.core.message import (
    Message,
    MessageRole,
)
from miniagent.core.state import State, WorkflowStatus


class TestState:
    """Tests for the State model."""

    def test_default_state(self) -> None:
        """A fresh state should have sensible defaults."""
        state = State()
        assert state.status == WorkflowStatus.PENDING
        assert state.messages == []
        assert state.variables == {}
        assert state.step_count == 0
        assert state.max_steps == 100
        assert state.current_node is None

    def test_add_message_returns_new_state(self) -> None:
        """add_message should return a new State, not mutate."""
        state = State()
        msg = Message(role=MessageRole.USER, content="Hi")
        new_state = state.add_message(msg)
        assert len(new_state.messages) == 1
        assert len(state.messages) == 0  # Original unchanged

    def test_increment_step(self) -> None:
        """increment_step should increase step_count by 1."""
        state = State(step_count=0)
        new_state = state.increment_step()
        assert new_state.step_count == 1
        assert state.step_count == 0  # Original unchanged

    def test_increment_step_at_max_raises(self) -> None:
        """increment_step at max_steps should raise ValueError."""
        state = State(step_count=5, max_steps=5)
        with pytest.raises(ValueError, match="maximum allowed steps"):
            state.increment_step()

    def test_set_variable(self) -> None:
        """set_variable should return a new state with the variable."""
        state = State()
        new_state = state.set_variable("key", "value")
        assert new_state.variables["key"] == "value"
        assert "key" not in state.variables

    def test_negative_step_count_rejected(self) -> None:
        """Negative step_count should be rejected."""
        with pytest.raises(ValidationError):
            State(step_count=-1)

    def test_zero_max_steps_rejected(self) -> None:
        """max_steps must be at least 1."""
        with pytest.raises(ValidationError):
            State(max_steps=0)
