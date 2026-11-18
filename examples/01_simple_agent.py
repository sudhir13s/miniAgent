"""Example 01 — simple agent: one agent, no tools, one question.

What this shows:
  1. Build an agent profile in code (or load it from YAML)
  2. Create an Agent (any model LiteLLM supports, e.g. OpenAI, Gemini, Claude)
  3. Build a State with a user message
  4. Call agent.run(state) → get back the response

Run:
    export OPENAI_API_KEY="sk-..."
    uv run python examples/01_simple_agent.py
"""

import asyncio

from console import print_conversation
from settings import example_model

from miniagent import Agent, AgentProfile, Message, MessageRole, ModelConfig, State
from miniagent.llm import LiteLLMProvider

QUESTION = "What is the capital of France, and why is it historically significant?"


async def main() -> None:
    """Ask a single question and print the answer."""
    # ── 1. Define the agent's profile (name, model, system prompt) ──────────
    profile = AgentProfile(
        name="assistant",
        system_prompt="You are a helpful, concise AI assistant.",
        model=ModelConfig(
            id=example_model(),
            temperature=0.7,
            max_tokens=512,
        ),
        max_iterations=3,
    )

    # ── 2. Create the agent with a provider ─────────────────────────────────
    #   LiteLLMProvider reaches any supported model (OpenAI, Gemini, Claude,
    #   Groq, Bedrock, …) through one interface.
    agent = Agent(profile=profile, llm_provider=LiteLLMProvider())

    # ── 3. Create a State and add the user's question ───────────────────────
    #   State's helper methods never change it in place — each returns a new State.
    state = State().add_message(Message(role=MessageRole.USER, content=QUESTION))

    # ── 4. Run the agent and print the result ───────────────────────────────
    print(f"Running agent: {profile.name} ({profile.model.id})\n")
    final_state = await agent.run(state)
    print_conversation(final_state)
    print(f"Steps used: {final_state.step_count}")


if __name__ == "__main__":
    asyncio.run(main())
