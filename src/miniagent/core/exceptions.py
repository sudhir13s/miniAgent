"""Exception hierarchy for miniAgent.

Every framework error is a ``MiniAgentError`` carrying a message and,
when it wraps another failure, the ``cause``.
"""

from __future__ import annotations


class MiniAgentError(Exception):
    """Base exception for all miniAgent errors.

    All framework-specific exceptions inherit from this class, enabling
    callers to catch the entire family with a single except clause when
    appropriate.

    Args:
        message: Human-readable description of the error.
        cause: Optional underlying exception that triggered this error.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize with a message and optional causal exception."""
        self.message = message
        self.cause = cause
        super().__init__(message)

    def __str__(self) -> str:
        """Return a formatted error string including the causal chain."""
        base = f"{self.__class__.__name__}: {self.message}"
        if self.cause is not None:
            base += f" | Caused by: {self.cause!r}"
        return base


class ConfigurationError(MiniAgentError):
    """Raised when YAML configuration is invalid or missing.

    Examples:
        - A required field is absent from the YAML file.
        - A model parameter value is out of its valid range.
        - The YAML file cannot be parsed.
    """


class AgentExecutionError(MiniAgentError):
    """Raised when an agent fails during execution.

    Examples:
        - The underlying LLM returns an unexpected response format.
        - A timeout occurs while waiting for the LLM response.
    """


class ToolInvocationError(MiniAgentError):
    """Raised when a tool call fails.

    Examples:
        - The tool receives arguments that fail validation.
        - An external API called by the tool returns an error.
    """


class WorkflowExecutionError(MiniAgentError):
    """Raised when a workflow encounters an unrecoverable error.

    Examples:
        - A graph edge routes to a node that is not registered.
        - The workflow exceeds its maximum step limit.
    """


class SessionStoreError(MiniAgentError):
    """Raised when conversation history cannot be loaded or saved.

    Examples:
        - A session id contains characters outside the allowed set.
        - The session database cannot be opened, read, or written.
    """


class LLMProviderError(MiniAgentError):
    """Raised when the LLM provider (LiteLLM) fails to complete a request.

    Examples:
        - Rate limit exceeded.
        - Authentication failure (invalid API key).
        - Request timeout.
        - Unsupported model string.
    """
