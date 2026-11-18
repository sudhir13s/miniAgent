"""Convert between miniAgent messages and the wire format LiteLLM speaks.

Pure functions, shared by every ``LiteLLMProvider`` call: requests turn
``Message`` and ``ToolSchema`` objects into wire-format dicts, and replies
(streamed or not) turn back into an AI ``Message`` plus ``ToolCall`` objects.
Wire names come from ``litellm_spec``; message fields from ``core.message``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from miniagent.core.exceptions import LLMProviderError
from miniagent.core.message import Message, MessageRole, ToolCall
from miniagent.llm.litellm_spec import (
    TOOL_TYPE_FUNCTION,
    MessageField,
    ReplyAttribute,
    RequestParam,
    ToolCallField,
    ToolDefinitionField,
    WireRole,
)

if TYPE_CHECKING:
    from litellm.types.utils import Message as LiteLLMMessage

    from miniagent.profile.model_config import ModelConfig
    from miniagent.tool.base import ToolSchema

_WIRE_ROLES: dict[MessageRole, str] = {
    MessageRole.SYSTEM: WireRole.SYSTEM,
    MessageRole.USER: WireRole.USER,
    MessageRole.AI: WireRole.ASSISTANT,
    MessageRole.TOOL: WireRole.TOOL,
}

# Message.content must be non-empty, so a tool-call-only AI turn stores this
# placeholder; it is sent back to the model as empty content.
TOOL_CALL_PLACEHOLDER = "[tool call]"


def to_litellm_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert the conversation to wire-format message dicts."""
    return [_to_wire_message(message) for message in messages]


def to_litellm_tools(tools: Sequence[ToolSchema]) -> list[dict[str, Any]]:
    """Convert tool schemas to wire-format function definitions."""
    return [
        {
            ToolDefinitionField.TYPE: TOOL_TYPE_FUNCTION,
            ToolDefinitionField.FUNCTION: {
                ToolDefinitionField.NAME: tool.name,
                ToolDefinitionField.DESCRIPTION: tool.description,
                ToolDefinitionField.PARAMETERS: tool.parameters,
            },
        }
        for tool in tools
    ]


def to_litellm_parameters(model_config: ModelConfig) -> dict[str, Any]:
    """Convert model settings to ``litellm.acompletion`` keyword arguments.

    ``extra_params`` is applied last, so it can add any LiteLLM option
    (``thinking``, ``response_format``, ``num_retries``, ...) or override a
    standard one.
    """
    parameters: dict[str, Any] = {
        RequestParam.MODEL: model_config.id,
        RequestParam.TEMPERATURE: model_config.temperature,
        RequestParam.MAX_TOKENS: model_config.max_tokens,
        RequestParam.TOP_P: model_config.top_p,
    }
    if model_config.top_k is not None:
        parameters[RequestParam.TOP_K] = model_config.top_k
    parameters.update(model_config.extra_params)
    return parameters


def to_ai_turn(response_message: LiteLLMMessage, model_id: str) -> tuple[Message, list[ToolCall]]:
    """Turn a LiteLLM reply message into an AI ``Message`` and its tool calls.

    Args:
        response_message: ``response.choices[0].message`` from LiteLLM.
        model_id: The model that replied, recorded on the message.

    Returns:
        The AI message, which also carries the tool calls, and the tool calls.

    Raises:
        LLMProviderError: If the reply has neither text nor tool calls, or
            a tool call does not name its tool.
    """
    tool_calls = parse_tool_calls(response_message)
    content: str = response_message.content or ""
    if not content and not tool_calls:
        raise LLMProviderError(
            f"Model '{model_id}' returned an empty reply: no text and no tool calls."
        )
    message = Message(
        role=MessageRole.AI,
        content=content or TOOL_CALL_PLACEHOLDER,
        tool_calls=tool_calls,
        model=model_id,
        reasoning=_reasoning(response_message),
        thinking_blocks=_thinking_blocks(response_message),
    )
    return message, tool_calls


def parse_tool_calls(response_message: LiteLLMMessage) -> list[ToolCall]:
    """Convert LiteLLM tool calls to ``ToolCall`` objects.

    Malformed JSON arguments become ``{}``, so the tool's own validation
    reports the problem to the model.

    Raises:
        LLMProviderError: If a tool call does not name its tool.
    """
    tool_calls: list[ToolCall] = []
    for raw_call in response_message.tool_calls or []:
        tool_name = raw_call.function.name
        if not tool_name:
            raise LLMProviderError("The model requested a tool call without naming the tool.")
        try:
            arguments: dict[str, Any] = json.loads(raw_call.function.arguments)
        except json.JSONDecodeError:
            arguments = {}
        tool_calls.append(ToolCall(id=raw_call.id or "", tool_name=tool_name, arguments=arguments))
    return tool_calls


def _reasoning(response_message: LiteLLMMessage) -> str | None:
    """Return the reply's readable reasoning (e.g. Claude extended thinking), if any.

    LiteLLM removes the attribute from replies that have no reasoning, so it
    is read with ``getattr``.
    """
    reasoning = getattr(response_message, ReplyAttribute.REASONING_CONTENT, None)
    return reasoning if isinstance(reasoning, str) and reasoning else None


def _thinking_blocks(response_message: LiteLLMMessage) -> list[dict[str, Any]]:
    """Return the reply's signed thinking blocks, which tool-use turns must send back."""
    blocks = getattr(response_message, ReplyAttribute.THINKING_BLOCKS, None)
    return [dict(block) for block in blocks] if isinstance(blocks, list) else []


def _to_wire_message(message: Message) -> dict[str, Any]:
    """Convert one message, re-attaching tool calls and thinking blocks to AI turns.

    Providers reject a tool result whose request is missing, so an AI turn
    that requested tools carries its tool calls back to the model.
    """
    entry: dict[str, Any] = {
        MessageField.ROLE: _WIRE_ROLES[message.role],
        MessageField.CONTENT: message.content,
    }
    if message.role == MessageRole.TOOL and message.tool_call_id:
        entry[MessageField.TOOL_CALL_ID] = message.tool_call_id
    if message.tool_calls:
        entry[MessageField.TOOL_CALLS] = [_to_wire_tool_call(call) for call in message.tool_calls]
        if message.content == TOOL_CALL_PLACEHOLDER:
            entry[MessageField.CONTENT] = None
    if message.thinking_blocks:
        entry[MessageField.THINKING_BLOCKS] = message.thinking_blocks
    return entry


def _to_wire_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    """Convert one requested tool call to its wire form."""
    return {
        ToolCallField.ID: tool_call.id,
        ToolCallField.TYPE: TOOL_TYPE_FUNCTION,
        ToolCallField.FUNCTION: {
            ToolCallField.NAME: tool_call.tool_name,
            ToolCallField.ARGUMENTS: json.dumps(tool_call.arguments),
        },
    }
