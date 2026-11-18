"""Tests for structured output: a validated Pydantic object as the agent's answer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field, ValidationError

from miniagent.agent import Agent, RunCompleted, build_agent
from miniagent.agent.structured_output import (
    OUTPUT_VARIABLE,
    correction_message,
    output_instructions,
    parse_output,
)
from miniagent.core.exceptions import AgentExecutionError
from miniagent.core.message import Message, MessageRole, ToolCall
from miniagent.core.state import State
from miniagent.profile.agent_profile import AgentProfile
from miniagent.profile.model_config import ModelConfig
from miniagent.tool.function_tool import tool
from tests.unit.llm.fakes import ScriptedProvider, text_turn, tool_turn


class Invoice(BaseModel):
    """An invoice extracted from text."""

    number: str = Field(description="Invoice number, e.g. 'INV-7'.")
    total: float


_VALID = '{"number": "INV-7", "total": 120.5}'


@tool
def lookup_total(number: str) -> float:
    """Look up an invoice total.

    Args:
        number: The invoice number.
    """
    return 120.5


def _profile(max_iterations: int = 5) -> AgentProfile:
    return AgentProfile(
        name="extractor",
        model=ModelConfig(id="openai/gpt-4o"),
        system_prompt="Extract invoices.",
        max_iterations=max_iterations,
    )


def _start() -> State:
    return State(messages=[Message(role=MessageRole.USER, content="Invoice INV-7, total 120.50")])


class TestOutputHelpers:
    def test_instructions_include_the_json_schema(self) -> None:
        instructions = output_instructions(Invoice)
        assert '"number"' in instructions
        assert "Invoice number, e.g. 'INV-7'." in instructions
        assert "only a JSON object" in instructions

    @pytest.mark.parametrize(
        "reply",
        [_VALID, f"```json\n{_VALID}\n```", f"```\n{_VALID}\n```", f"  {_VALID}\n"],
    )
    def test_parse_accepts_bare_or_fenced_json(self, reply: str) -> None:
        assert parse_output(reply, Invoice) == Invoice(number="INV-7", total=120.5)

    def test_parse_rejects_replies_that_do_not_match(self) -> None:
        with pytest.raises(ValidationError):
            parse_output('{"number": "INV-7"}', Invoice)

    def test_correction_names_what_was_wrong(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            parse_output('{"number": "INV-7"}', Invoice)
        message = correction_message(excinfo.value)
        assert message.role == MessageRole.USER
        assert "total: Field required" in message.content


class TestAgentWithOutputType:
    async def test_valid_reply_becomes_the_output_object(self) -> None:
        agent = Agent(_profile(), ScriptedProvider(text_turn(_VALID)), output_type=Invoice)

        final_state = await agent.run(_start())

        assert final_state.variables[OUTPUT_VARIABLE] == Invoice(number="INV-7", total=120.5)

    async def test_schema_is_added_to_the_system_prompt(self) -> None:
        provider = ScriptedProvider(text_turn(_VALID))
        await Agent(_profile(), provider, output_type=Invoice).run(_start())

        system_prompt = provider.received[0][0]
        assert system_prompt.role == MessageRole.SYSTEM
        assert system_prompt.content.startswith("Extract invoices.")
        assert '"total"' in system_prompt.content

    async def test_schema_is_added_to_a_system_prompt_already_in_the_state(self) -> None:
        provider = ScriptedProvider(text_turn(_VALID))
        state = State(
            messages=[Message(role=MessageRole.SYSTEM, content="Custom."), *_start().messages]
        )
        await Agent(_profile(), provider, output_type=Invoice).run(state)

        [system_prompt] = [m for m in provider.received[0] if m.role == MessageRole.SYSTEM]
        assert system_prompt.content.startswith("Custom.")
        assert '"total"' in system_prompt.content

    async def test_invalid_reply_is_sent_back_for_correction(self) -> None:
        provider = ScriptedProvider(text_turn('{"number": "INV-7"}'), text_turn(_VALID))
        agent = Agent(_profile(), provider, output_type=Invoice)

        final_state = await agent.run(_start())

        correction = provider.received[1][-1]
        assert correction.role == MessageRole.USER
        assert "total: Field required" in correction.content
        assert final_state.variables[OUTPUT_VARIABLE].total == 120.5

    async def test_fails_when_no_valid_reply_arrives_in_time(self) -> None:
        provider = ScriptedProvider(text_turn("not json"), text_turn("still not json"))
        agent = Agent(_profile(max_iterations=2), provider, output_type=Invoice)

        with pytest.raises(AgentExecutionError, match="valid Invoice"):
            await agent.run(_start())

    async def test_tools_can_run_before_the_structured_answer(self) -> None:
        call = ToolCall(id="c1", tool_name="lookup_total", arguments={"number": "INV-7"})
        provider = ScriptedProvider(tool_turn(call), text_turn(_VALID))
        agent = Agent(
            _profile(), provider, tools={"lookup_total": lookup_total}, output_type=Invoice
        )

        final_state = await agent.run(_start())

        assert final_state.variables[OUTPUT_VARIABLE].number == "INV-7"

    async def test_streamed_runs_complete_with_the_output_object(self) -> None:
        agent = Agent(
            _profile(),
            ScriptedProvider(text_turn('{"number": "INV-7", ', '"total": 120.5}')),
            output_type=Invoice,
        )

        events = [event async for event in agent.stream(_start())]

        completed = events[-1]
        assert isinstance(completed, RunCompleted)
        assert completed.state.variables[OUTPUT_VARIABLE] == Invoice(number="INV-7", total=120.5)

    async def test_without_output_type_text_replies_are_left_alone(self) -> None:
        final_state = await Agent(_profile(), ScriptedProvider(text_turn("hi"))).run(_start())
        assert OUTPUT_VARIABLE not in final_state.variables


class TestBuildAgentOutputType:
    def test_passes_the_output_type_to_the_agent(self) -> None:
        agent = build_agent(_profile(), llm_provider=MagicMock(), output_type=Invoice)
        assert agent.output_type is Invoice
