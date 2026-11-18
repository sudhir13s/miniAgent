"""Example 03 — sequential pipeline: researcher → writer.

The researcher gathers information, then the writer turns it into polished
prose. Each agent sees all messages from prior agents — the state flows
through the chain, growing with each agent's messages.

What this shows:
  1. SequentialWorkflow runs agents one after another
  2. State is passed through the chain; each agent sees all prior messages
  3. Agents can have different temperatures, prompts, and tool sets
  4. The final state contains the full conversation history

Run:
    export OPENAI_API_KEY="sk-..."
    uv run python examples/03_pipeline.py
"""

import asyncio

from console import DIVIDER, print_conversation
from settings import example_model

from miniagent import Agent, AgentProfile, Message, MessageRole, ModelConfig, State
from miniagent.llm import LiteLLMProvider
from miniagent.tool import WeatherTool, WebFetchTool
from miniagent.workflow import SequentialWorkflow

TOPIC = "What are the top 3 AI trends in 2025?"
MAX_TOOL_OUTPUT_CHARS = 200


async def main() -> None:
    """Run the researcher → writer pipeline on one topic."""
    provider = LiteLLMProvider()
    model = example_model()

    # ── 1. Define the researcher agent ──────────────────────────────────────
    researcher_profile = AgentProfile(
        name="researcher",
        system_prompt=(
            "You are a thorough research assistant. "
            "Use available tools to gather real information. "
            "Summarize what you find in clear bullet points."
        ),
        model=ModelConfig(id=model, temperature=0.3, max_tokens=1024),
        max_iterations=8,
    )
    researcher = Agent(
        profile=researcher_profile,
        llm_provider=provider,
        tools={"web_fetch": WebFetchTool(), "weather": WeatherTool()},
    )

    # ── 2. Define the writer agent ───────────────────────────────────────────
    writer_profile = AgentProfile(
        name="writer",
        system_prompt=(
            "You are a professional technical writer. "
            "You will receive research notes from a previous agent. "
            "Turn them into a polished, engaging 2-3 paragraph article. "
            "Do not use bullet points — write in flowing prose."
        ),
        model=ModelConfig(id=model, temperature=0.8, max_tokens=1024),
        max_iterations=3,
    )
    writer = Agent(profile=writer_profile, llm_provider=provider)

    # ── 3. Build the sequential workflow ────────────────────────────────────
    #   agent_order defines the execution sequence.
    workflow = SequentialWorkflow(name="research-and-write", agent_order=["researcher", "writer"])
    workflow.register_agent(researcher)
    workflow.register_agent(writer)

    # ── 4. Seed state with the user question and run ────────────────────────
    state = State().add_message(Message(role=MessageRole.USER, content=TOPIC))
    print("Pipeline: researcher → writer")
    print(f"Topic: {TOPIC}\n")
    print(DIVIDER)

    final_state = await workflow.run(state)
    print_conversation(final_state, max_tool_output_chars=MAX_TOOL_OUTPUT_CHARS)

    print(DIVIDER)
    print(f"Status: {final_state.status.value} | Steps: {final_state.step_count}")


if __name__ == "__main__":
    asyncio.run(main())
