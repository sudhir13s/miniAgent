"""Example 02 — agent with tools: the Reason + Act (ReAct) loop.

The agent reasons about a problem, calls tools to gather data, then
produces a final answer.

Tools used:
  - CalculatorTool  : safe arithmetic evaluation
  - TimestampTool   : current UTC time
  - WeatherTool     : real-time weather from wttr.in (no API key)

What this shows:
  1. Build a profile from a built-in template (TOOL_ASSISTANT), then let
     build_agent() attach the tools the template lists
  2. The agent automatically decides WHEN to call tools vs. answer directly
  3. Tool results are appended to State and visible in the next iteration
  4. The loop stops when the agent produces a text-only (no-tool) response

Run:
    export OPENAI_API_KEY="sk-..."
    uv run python examples/02_tool_agent.py
"""

import asyncio

from console import DIVIDER, print_conversation
from settings import example_model

from miniagent import AgentProfile, Message, MessageRole, ModelConfig, State, build_agent
from miniagent.profile.templates import TOOL_ASSISTANT

QUESTION = (
    "What is (123 * 456) + (789 / 3)? "
    "Also, what is the current UTC time, and what's the weather in Tokyo right now?"
)


async def main() -> None:
    """Ask a question that needs several tools and print the full ReAct trace."""
    # ── 1. Define the agent profile from a built-in template ────────────────
    #   The template supplies the system prompt, tool names, and iteration
    #   budget. Any field passed here explicitly overrides the template.
    profile = AgentProfile(
        name="smart-assistant",
        template=TOOL_ASSISTANT,
        model=ModelConfig(
            id=example_model(),
            temperature=0.3,
            max_tokens=1024,
        ),
    )

    # ── 2. Build the agent ───────────────────────────────────────────────────
    #   build_agent() looks up the profile's tool names (calculator, timestamp,
    #   weather) among the built-in tools and uses the default LiteLLM provider.
    agent = build_agent(profile)

    # ── 3. Run it and print the full conversation (shows the ReAct loop) ────
    state = State().add_message(Message(role=MessageRole.USER, content=QUESTION))
    print(f"Running agent: {profile.name}")
    print(f"Question: {QUESTION}\n")
    print(DIVIDER)

    final_state = await agent.run(state)
    print_conversation(final_state)

    print(DIVIDER)
    print(f"Steps used: {final_state.step_count}")


if __name__ == "__main__":
    asyncio.run(main())
