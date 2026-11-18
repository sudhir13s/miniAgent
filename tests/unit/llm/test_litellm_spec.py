"""Check the wire-format names against LiteLLM itself.

``litellm_spec`` is the only place that spells LiteLLM's names. These tests
compare it with the installed LiteLLM, so a name LiteLLM changes or one
mistyped here fails loudly instead of being sent and silently ignored.
"""

from __future__ import annotations

import inspect

import litellm
from litellm.types.utils import Message as LiteLLMMessage

from miniagent.llm.litellm_spec import (
    TOOL_CHOICE_AUTO,
    TOOL_TYPE_FUNCTION,
    MessageField,
    ReplyAttribute,
    RequestParam,
    ToolCallField,
    ToolDefinitionField,
    WireRole,
)


def _names(namespace: type) -> list[str]:
    return [value for name, value in vars(namespace).items() if not name.startswith("_")]


class TestRequestParameters:
    def test_should_all_be_parameters_litellm_accepts(self) -> None:
        accepted = inspect.signature(litellm.acompletion).parameters
        unknown = [name for name in _names(RequestParam) if name not in accepted]
        # top_k is not an OpenAI parameter; LiteLLM forwards it to vendors that have it.
        assert unknown == [RequestParam.TOP_K]
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values())

    def test_should_keep_the_openai_names(self) -> None:
        assert _names(RequestParam) == [
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "tools",
            "tool_choice",
            "stream",
        ]


class TestMessageFormat:
    def test_should_keep_the_openai_roles_and_keys(self) -> None:
        assert _names(WireRole) == ["system", "user", "assistant", "tool"]
        assert _names(MessageField) == [
            "role",
            "content",
            "tool_calls",
            "tool_call_id",
            "thinking_blocks",
        ]
        assert _names(ToolCallField) == ["id", "type", "function", "name", "arguments"]
        assert _names(ToolDefinitionField) == [
            "type",
            "function",
            "name",
            "description",
            "parameters",
        ]
        assert (TOOL_TYPE_FUNCTION, TOOL_CHOICE_AUTO) == ("function", "auto")

    def test_should_read_reasoning_from_attributes_litellm_sets(self) -> None:
        reply = LiteLLMMessage(
            content="Done.",
            reasoning_content="Think first.",
            thinking_blocks=[{"type": "thinking", "thinking": "Think first.", "signature": "s"}],
        )
        assert getattr(reply, ReplyAttribute.REASONING_CONTENT) == "Think first."
        assert getattr(reply, ReplyAttribute.THINKING_BLOCKS)[0]["signature"] == "s"
