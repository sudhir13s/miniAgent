"""Unit tests for the streaming event schema: the names and fields clients rely on."""

from __future__ import annotations

from miniagent.api.stream_events import (
    ErrorPayload,
    StreamEventName,
    TextPayload,
    ToolCallPayload,
    ToolResultPayload,
    WorkflowErrorPayload,
    WorkflowStartedPayload,
)


class TestEventNames:
    def test_should_keep_the_documented_event_names(self) -> None:
        assert [name.value for name in StreamEventName] == [
            "text",
            "tool_call",
            "tool_result",
            "started",
            "completed",
            "error",
        ]


class TestPayloadFields:
    def test_should_keep_the_documented_agent_event_fields(self) -> None:
        assert TextPayload(delta="Hi").model_dump() == {"delta": "Hi"}
        assert ToolCallPayload(id="c1", name="add", arguments={"a": 1}).model_dump() == {
            "id": "c1",
            "name": "add",
            "arguments": {"a": 1},
        }
        result = ToolResultPayload(id="c1", name="add", content="2", is_error=False)
        assert result.model_dump() == {"id": "c1", "name": "add", "content": "2", "is_error": False}
        assert ErrorPayload(detail="boom").model_dump() == {"detail": "boom"}

    def test_should_name_the_workflow_in_workflow_events(self) -> None:
        assert WorkflowStartedPayload(workflow_name="pipeline").model_dump() == {
            "workflow_name": "pipeline"
        }
        failure = WorkflowErrorPayload(workflow_name="pipeline", detail="boom")
        assert failure.model_dump() == {"detail": "boom", "workflow_name": "pipeline"}
