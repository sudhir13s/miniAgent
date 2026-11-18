"""The wire format LiteLLM speaks: every name miniAgent sends to it or reads back.

LiteLLM uses the OpenAI chat-completion format for every vendor. The names in
that format are defined here, once; ``litellm_format`` and ``litellm_provider``
use these definitions and never spell a wire name themselves. If LiteLLM's
format changes, this is the only file to edit.

Reference: https://docs.litellm.ai/docs/completion/input
"""

from __future__ import annotations

from typing import Final


class RequestParam:
    """Keyword arguments of ``litellm.acompletion``."""

    MODEL: Final = "model"
    MESSAGES: Final = "messages"
    TEMPERATURE: Final = "temperature"
    MAX_TOKENS: Final = "max_tokens"
    TOP_P: Final = "top_p"
    # Not an OpenAI parameter: LiteLLM forwards it to the vendors that support it.
    TOP_K: Final = "top_k"
    TOOLS: Final = "tools"
    TOOL_CHOICE: Final = "tool_choice"
    STREAM: Final = "stream"


class WireRole:
    """Values of a wire message's ``role``."""

    SYSTEM: Final = "system"
    USER: Final = "user"
    ASSISTANT: Final = "assistant"
    TOOL: Final = "tool"


class MessageField:
    """Keys of one message in the ``messages`` list."""

    ROLE: Final = "role"
    CONTENT: Final = "content"
    TOOL_CALLS: Final = "tool_calls"
    TOOL_CALL_ID: Final = "tool_call_id"
    THINKING_BLOCKS: Final = "thinking_blocks"


class ToolCallField:
    """Keys of one entry in an assistant message's ``tool_calls``."""

    ID: Final = "id"
    TYPE: Final = "type"
    FUNCTION: Final = "function"
    NAME: Final = "name"
    ARGUMENTS: Final = "arguments"


class ToolDefinitionField:
    """Keys of one entry in the request's ``tools`` list."""

    TYPE: Final = "type"
    FUNCTION: Final = "function"
    NAME: Final = "name"
    DESCRIPTION: Final = "description"
    PARAMETERS: Final = "parameters"


class ReplyAttribute:
    """Optional attributes of a reply message that carry model reasoning."""

    REASONING_CONTENT: Final = "reasoning_content"
    THINKING_BLOCKS: Final = "thinking_blocks"


TOOL_TYPE_FUNCTION: Final = "function"
"""The only tool type miniAgent offers and accepts."""

TOOL_CHOICE_AUTO: Final = "auto"
"""Lets the model decide whether to call a tool."""
