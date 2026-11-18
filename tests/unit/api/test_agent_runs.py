"""Unit tests for the helpers the agent run endpoints share."""

from __future__ import annotations

import pytest

from miniagent.agent.events import RunCompleted, TextDelta, ToolCallFinished, ToolCallStarted
from miniagent.api.agent_runs import NO_RESPONSE, RunSession, final_response, progress_event
from miniagent.core.message import Message, MessageRole, ToolCall
from miniagent.core.state import State
from miniagent.memory.session_store import InMemorySessionStore


def _message(role: MessageRole, content: str) -> Message:
    return Message(role=role, content=content)


class TestRunSession:
    async def test_should_start_from_system_prompt_history_and_new_message(self) -> None:
        store = InMemorySessionStore()
        await store.append("s1", [_message(MessageRole.USER, "earlier")])

        state = await RunSession(store, "s1").start("Be brief.", "now", {"k": "v"})

        assert [m.content for m in state.messages] == ["Be brief.", "earlier", "now"]
        assert state.variables == {"k": "v"}

    async def test_should_save_only_what_the_run_added(self) -> None:
        store = InMemorySessionStore()
        session = RunSession(store, "s1")
        state = await session.start("Be brief.", "hi", {})

        await session.finish(state.add_message(_message(MessageRole.AI, "hello")))

        assert [m.content for m in await store.load("s1")] == ["hi", "hello"]

    async def test_should_neither_load_nor_save_without_a_session_id(self) -> None:
        store = InMemorySessionStore()
        session = RunSession(store, None)
        state = await session.start("Be brief.", "hi", {})
        await session.finish(state)
        assert [m.content for m in state.messages] == ["Be brief.", "hi"]


class TestFinalResponse:
    def test_should_return_the_last_ai_message(self) -> None:
        state = State(
            messages=[_message(MessageRole.AI, "first"), _message(MessageRole.AI, "last")]
        )
        assert final_response(state) == "last"

    def test_should_skip_a_tool_request_the_run_ended_on(self) -> None:
        call = ToolCall(id="c1", tool_name="echo")
        request = Message(role=MessageRole.AI, content="[tool call]", tool_calls=[call])
        state = State(messages=[_message(MessageRole.AI, "partial answer"), request])
        assert final_response(state) == "partial answer"

    def test_should_return_a_placeholder_without_ai_messages(self) -> None:
        assert final_response(State()) == NO_RESPONSE


class TestProgressEvent:
    def test_should_map_text(self) -> None:
        name, payload = progress_event(TextDelta("Hi"))
        assert (name.value, payload.model_dump()) == ("text", {"delta": "Hi"})

    def test_should_map_tool_calls(self) -> None:
        call = ToolCall(id="c1", tool_name="add", arguments={"a": 1})
        name, payload = progress_event(ToolCallStarted(call))
        assert name.value == "tool_call"
        assert payload.model_dump() == {"id": "c1", "name": "add", "arguments": {"a": 1}}

    def test_should_map_tool_results(self) -> None:
        result = Message(
            role=MessageRole.TOOL,
            content="boom",
            tool_call_id="c1",
            tool_name="add",
            is_error=True,
        )
        name, payload = progress_event(ToolCallFinished(result))
        assert name.value == "tool_result"
        assert payload.model_dump() == {
            "id": "c1",
            "name": "add",
            "content": "boom",
            "is_error": True,
        }

    def test_should_refuse_run_completed(self) -> None:
        with pytest.raises(ValueError, match="RunCompleted"):
            progress_event(RunCompleted(State()))
