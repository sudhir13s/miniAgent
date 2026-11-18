"""Invoke the tools a model requested and turn every outcome into a TOOL message.

A failing tool never aborts the agent's run: an unknown tool, invalid
arguments, an exception or a timeout all come back as error results the model
can react to. Cancellation still propagates.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from miniagent.core.exceptions import MiniAgentError
from miniagent.core.message import Message, MessageRole, ToolCall

if TYPE_CHECKING:
    from miniagent.tool.base import BaseTool

logger = logging.getLogger(__name__)

# A message needs content, and the model should know the tool ran but said nothing.
_NO_OUTPUT = "(no output)"


async def invoke_tool_calls(
    tool_calls: Sequence[ToolCall], tools: Mapping[str, BaseTool]
) -> list[Message]:
    """Invoke every tool call concurrently.

    Args:
        tool_calls: The calls the model requested.
        tools: The tools available for this run, by name.

    Returns:
        One TOOL message per call, in the order the model requested them.
    """
    return list(await asyncio.gather(*(_invoke_tool(call, tools) for call in tool_calls)))


async def _invoke_tool(tool_call: ToolCall, tools: Mapping[str, BaseTool]) -> Message:
    """Invoke one tool and return its TOOL message, never raising for tool failures."""
    tool_name = tool_call.tool_name
    tool = tools.get(tool_name)
    if tool is None:
        logger.warning("Model requested unknown tool '%s'", tool_name)
        return _tool_message(
            tool_call, f"Tool '{tool_name}' is not registered with this agent.", is_error=True
        )

    try:
        output = await asyncio.wait_for(
            tool.invoke(**tool_call.arguments), timeout=tool.timeout_seconds
        )
    except asyncio.TimeoutError:
        logger.warning("Tool '%s' timed out after %ss", tool_name, tool.timeout_seconds)
        return _tool_message(
            tool_call,
            f"Tool '{tool_name}' timed out after {tool.timeout_seconds:g} seconds.",
            is_error=True,
        )
    except Exception as exc:
        logger.warning("Tool '%s' failed", tool_name, exc_info=True)
        return _tool_message(
            tool_call, f"Tool '{tool_name}' failed: {_describe_error(exc)}", is_error=True
        )
    return _tool_message(tool_call, output or _NO_OUTPUT, is_error=False)


def _tool_message(tool_call: ToolCall, content: str, *, is_error: bool) -> Message:
    """Build the TOOL-role message that answers ``tool_call``."""
    return Message(
        role=MessageRole.TOOL,
        content=content,
        tool_call_id=tool_call.id,
        tool_name=tool_call.tool_name,
        is_error=is_error,
    )


def _describe_error(exc: Exception) -> str:
    """Describe a tool failure for the model.

    Framework errors carry a message written for the model; anything else is
    reported with its type so the model can tell a bug from bad input.
    """
    if isinstance(exc, MiniAgentError):
        return exc.message
    return f"{type(exc).__name__}: {exc}"
