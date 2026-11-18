"""Comprehensive tests for the custom exception hierarchy.

Validates exception chaining, string formatting, and type relationships.
"""

from __future__ import annotations

from miniagent.core.exceptions import (
    AgentExecutionError,
    ConfigurationError,
    MiniAgentError,
    SessionStoreError,
    ToolInvocationError,
    WorkflowExecutionError,
)


class TestExceptions:
    """Tests for the exception hierarchy."""

    def test_base_exception(self) -> None:
        """MiniAgentError should store message."""
        exc = MiniAgentError("test error")
        assert exc.message == "test error"
        assert exc.cause is None
        assert "MiniAgentError: test error" in str(exc)

    def test_exception_with_cause(self) -> None:
        """Should chain an underlying cause."""
        cause = ValueError("original error")
        exc = AgentExecutionError("wrapped error", cause=cause)
        assert exc.cause is cause
        assert "Caused by" in str(exc)
        assert "ValueError" in str(exc)

    def test_all_exceptions_inherit_from_base(self) -> None:
        """All custom exceptions should be MiniAgentError subclasses."""
        exceptions = [
            ConfigurationError("test"),
            AgentExecutionError("test"),
            ToolInvocationError("test"),
            WorkflowExecutionError("test"),
            SessionStoreError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, MiniAgentError)
            assert isinstance(exc, Exception)

    def test_exceptions_are_catchable_by_base(self) -> None:
        """Catching MiniAgentError should catch all subtypes."""
        with __import__("pytest").raises(MiniAgentError):
            raise ConfigurationError("config fail")

    def test_exception_without_cause_str(self) -> None:
        """String format should not include 'Caused by' when no cause."""
        exc = ToolInvocationError("tool broke")
        assert "Caused by" not in str(exc)
