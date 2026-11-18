"""Tests for Message and ToolCall."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miniagent.core.message import Message, MessageRole, ToolCall


class TestMessage:
    """Tests for the Message model."""

    def test_create_user_message(self) -> None:
        """A user message should be created with required fields."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.id  # UUID auto-generated
        assert msg.timestamp
        assert msg.metadata == {}

    def test_create_system_message(self) -> None:
        """A system message should be created correctly."""
        msg = Message(role=MessageRole.SYSTEM, content="Be helpful.")
        assert msg.role == MessageRole.SYSTEM

    def test_create_ai_message(self) -> None:
        """An AI message should be created correctly."""
        msg = Message(role=MessageRole.AI, content="Sure!")
        assert msg.role == MessageRole.AI

    def test_tool_message_names_the_call_it_answers(self) -> None:
        msg = Message(
            role=MessageRole.TOOL,
            content="Result: 42",
            tool_call_id="tc-123",
            tool_name="calculator",
            is_error=True,
        )
        assert (msg.tool_call_id, msg.tool_name, msg.is_error) == ("tc-123", "calculator", True)

    def test_ai_message_carries_its_tool_calls_and_reasoning(self) -> None:
        call = ToolCall(id="tc-1", tool_name="calculator", arguments={"expression": "2+2"})
        msg = Message(
            role=MessageRole.AI,
            content="[tool call]",
            tool_calls=[call],
            model="openai/gpt-4o",
            reasoning="Add the numbers.",
        )
        assert msg.tool_calls == [call]
        assert (msg.model, msg.reasoning) == ("openai/gpt-4o", "Add the numbers.")

    def test_schema_fields_default_to_empty(self) -> None:
        msg = Message(role=MessageRole.USER, content="Hello")
        assert (msg.tool_calls, msg.thinking_blocks) == ([], [])
        assert (msg.tool_call_id, msg.tool_name, msg.model, msg.reasoning) == (None,) * 4
        assert msg.is_error is False

    def test_message_survives_a_json_round_trip(self) -> None:
        call = ToolCall(id="tc-1", tool_name="echo", arguments={"text": "x"})
        msg = Message(role=MessageRole.AI, content="[tool call]", tool_calls=[call])
        assert Message.model_validate_json(msg.model_dump_json()) == msg

    def test_empty_content_rejected(self) -> None:
        """An empty content string should be rejected."""
        with pytest.raises(ValidationError):
            Message(role=MessageRole.USER, content="")

    def test_missing_role_rejected(self) -> None:
        """Missing role field should be rejected."""
        with pytest.raises(ValidationError):
            Message(content="Hello")  # type: ignore[call-arg]

    def test_invalid_role_rejected(self) -> None:
        """An invalid role string should be rejected."""
        with pytest.raises(ValidationError):
            Message(role="invalid", content="Hello")  # type: ignore[arg-type]

    def test_message_is_frozen(self) -> None:
        """Messages should be immutable."""
        msg = Message(role=MessageRole.USER, content="Hello")
        with pytest.raises(ValidationError):
            msg.content = "Modified"  # type: ignore[misc]

    def test_message_with_metadata(self) -> None:
        """A message should accept metadata dictionary."""
        msg = Message(
            role=MessageRole.AI,
            content="Response",
            metadata={"source": "test", "score": 0.95},
        )
        assert msg.metadata["source"] == "test"
        assert msg.metadata["score"] == 0.95


class TestToolCall:
    """Tests for the ToolCall model."""

    def test_create_tool_call(self) -> None:
        """A tool call should be created with name and arguments."""
        tc = ToolCall(tool_name="search", arguments={"query": "AI"})
        assert tc.tool_name == "search"
        assert tc.arguments == {"query": "AI"}
        assert tc.id  # Auto-generated

    def test_empty_tool_name_rejected(self) -> None:
        """An empty tool name should be rejected."""
        with pytest.raises(ValidationError):
            ToolCall(tool_name="")

    def test_tool_call_is_frozen(self) -> None:
        """ToolCall should be immutable."""
        tc = ToolCall(tool_name="search")
        with pytest.raises(ValidationError):
            tc.tool_name = "other"  # type: ignore[misc]
