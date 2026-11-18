"""Unit tests for invoke_tool_calls: every outcome becomes a TOOL message."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from miniagent.agent.tool_invocation import invoke_tool_calls
from miniagent.core.exceptions import ToolInvocationError
from miniagent.core.message import MessageRole, ToolCall
from miniagent.tool.base import BaseTool, ToolSchema


class _ScriptedTool(BaseTool):
    """Tool whose behaviour is chosen per test: reply, raise, or sleep."""

    def __init__(
        self,
        name: str,
        *,
        reply: str = "ok",
        error: Exception | None = None,
        delay_seconds: float = 0.0,
        timeout_seconds: float | None = 1.0,
    ) -> None:
        super().__init__(ToolSchema(name=name, description=f"The {name} tool.", parameters={}))
        self._reply = reply
        self._error = error
        self._delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds

    async def invoke(self, **kwargs: Any) -> str:
        await asyncio.sleep(self._delay_seconds)
        if self._error is not None:
            raise self._error
        return f"{self._reply} {kwargs}" if kwargs else self._reply


def _call(tool_name: str, call_id: str = "c1", **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, tool_name=tool_name, arguments=arguments)


class TestInvokeToolCalls:
    async def test_should_return_result_linked_to_its_call(self) -> None:
        tools = {"echo": _ScriptedTool("echo", reply="hi")}
        [message] = await invoke_tool_calls([_call("echo", text="x")], tools)
        assert message.role == MessageRole.TOOL
        assert message.content == "hi {'text': 'x'}"
        assert (message.tool_name, message.tool_call_id, message.is_error) == ("echo", "c1", False)
        assert message.metadata == {}

    async def test_should_say_so_when_a_tool_returns_nothing(self) -> None:
        tools = {"quiet": _ScriptedTool("quiet", reply="")}
        [message] = await invoke_tool_calls([_call("quiet")], tools)
        assert message.content == "(no output)"
        assert message.is_error is False

    async def test_should_report_unknown_tool_as_error(self) -> None:
        [message] = await invoke_tool_calls([_call("missing")], {})
        assert message.is_error is True
        assert "not registered" in message.content

    async def test_should_pass_framework_error_message_to_model(self) -> None:
        tools = {"bad": _ScriptedTool("bad", error=ToolInvocationError("Invalid date"))}
        [message] = await invoke_tool_calls([_call("bad")], tools)
        assert message.is_error is True
        assert message.content == "Tool 'bad' failed: Invalid date"

    async def test_should_name_exception_type_for_unexpected_errors(self) -> None:
        tools = {"buggy": _ScriptedTool("buggy", error=KeyError("missing_key"))}
        [message] = await invoke_tool_calls([_call("buggy")], tools)
        assert message.content == "Tool 'buggy' failed: KeyError: 'missing_key'"

    async def test_should_report_timeout_as_error(self) -> None:
        tools = {"slow": _ScriptedTool("slow", delay_seconds=1, timeout_seconds=0.01)}
        [message] = await invoke_tool_calls([_call("slow")], tools)
        assert message.is_error is True
        assert message.content == "Tool 'slow' timed out after 0.01 seconds."

    async def test_should_keep_call_order_when_later_calls_finish_first(self) -> None:
        tools = {
            "slow": _ScriptedTool("slow", reply="first", delay_seconds=0.05),
            "fast": _ScriptedTool("fast", reply="second"),
        }
        messages = await invoke_tool_calls([_call("slow", "c1"), _call("fast", "c2")], tools)
        assert [message.content for message in messages] == ["first", "second"]
        assert [message.tool_call_id for message in messages] == ["c1", "c2"]

    async def test_should_run_calls_concurrently(self) -> None:
        tools = {name: _ScriptedTool(name, delay_seconds=0.2) for name in ("a", "b", "c")}
        calls = [_call(name, f"c-{name}") for name in tools]
        loop = asyncio.get_running_loop()
        started = loop.time()
        await invoke_tool_calls(calls, tools)
        assert loop.time() - started < 0.5

    async def test_should_propagate_cancellation(self) -> None:
        tools = {"slow": _ScriptedTool("slow", delay_seconds=10, timeout_seconds=None)}
        task = asyncio.ensure_future(invoke_tool_calls([_call("slow")], tools))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
