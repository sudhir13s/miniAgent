"""The README quick start, run end to end through LiteLLM with an offline reply.

``mock_response`` is LiteLLM's own offline mode: the request goes through the
real provider code, and LiteLLM answers without calling a vendor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from miniagent import AgentProfile, Message, MessageRole, State, build_agent
from miniagent.agent import RunCompleted, TextDelta
from miniagent.core.state import WorkflowStatus

_QUICK_START_CONFIG = """\
agents:
  - name: assistant
    system_prompt: "You are a concise, helpful assistant."
    model: {id: "openai/gpt-4o-mini", extra_params: {mock_response: "17 * 23 = 391"}}
    tools: [calculator, timestamp]
"""


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "agents.yaml"
    path.write_text(_QUICK_START_CONFIG, encoding="utf-8")
    return path


def _question() -> State:
    return State(messages=[Message(role=MessageRole.USER, content="What is 17 * 23?")])


class TestReadmeQuickStart:
    async def test_should_answer_from_a_yaml_defined_agent(self, config_path: Path) -> None:
        agent = build_agent(AgentProfile.from_yaml(config_path, name="assistant"))

        final_state = await agent.run(_question())

        assert final_state.messages[-1].content == "17 * 23 = 391"
        assert final_state.status == WorkflowStatus.COMPLETED
        assert sorted(agent.tools) == ["calculator", "timestamp"]

    async def test_should_stream_the_same_answer(self, config_path: Path) -> None:
        agent = build_agent(AgentProfile.from_yaml(config_path, name="assistant"))

        events = [event async for event in agent.stream(_question())]

        text = "".join(event.text for event in events if isinstance(event, TextDelta))
        assert text == "17 * 23 = 391"
        assert isinstance(events[-1], RunCompleted)
        assert events[-1].state.messages[-1].content == text
