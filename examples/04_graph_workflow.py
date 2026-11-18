"""Example 04 — graph workflow with conditional routing.

The editor can send the writer back for revisions; the routing decision is
based on the editor's verdict in its last message.

  writer → editor → (if approved) → END
                  ↘ (if needs revision) → writer (loop back)

What this shows:
  1. GraphWorkflow: node-and-edge execution with dynamic routing
  2. Conditional edges: router functions inspect State to decide the next node
  3. END_NODE sentinel terminates the graph
  4. max_steps prevents infinite loops

Run:
    export OPENAI_API_KEY="sk-..."
    uv run python examples/04_graph_workflow.py
"""

import asyncio

from console import DIVIDER, print_conversation
from settings import example_model

from miniagent import Agent, AgentProfile, Message, MessageRole, ModelConfig, State
from miniagent.llm import LiteLLMProvider
from miniagent.workflow import END_NODE, GraphWorkflow

TOPIC = "Why is Python the most popular language for AI development?"
APPROVED_VERDICT = "APPROVED"
REVISION_VERDICT = "NEEDS REVISION"
MAX_STEPS = 50


def route_after_editor(state: State) -> str:
    """Send the draft back to the writer on a revision verdict; otherwise finish."""
    ai_messages = [message for message in state.messages if message.role == MessageRole.AI]
    if not ai_messages:
        return END_NODE
    verdict = ai_messages[-1].content.upper()
    if REVISION_VERDICT in verdict and APPROVED_VERDICT not in verdict:
        return "writer"
    return END_NODE


async def main() -> None:
    """Run the writer ↔ editor loop on one topic."""
    provider = LiteLLMProvider()
    model = example_model()

    # ── 1. Writer agent ──────────────────────────────────────────────────────
    writer_profile = AgentProfile(
        name="writer",
        system_prompt=(
            "You are a writer. Write a short 2-sentence summary on the topic. "
            "If you have received feedback, incorporate it and improve the text. "
            "After writing, add exactly this line: QUALITY: [a number 1-10]"
        ),
        model=ModelConfig(id=model, temperature=0.7, max_tokens=256),
        max_iterations=2,
    )
    writer = Agent(profile=writer_profile, llm_provider=provider)

    # ── 2. Editor agent ──────────────────────────────────────────────────────
    editor_profile = AgentProfile(
        name="editor",
        system_prompt=(
            "You are an editor reviewing a piece of writing. "
            "Rate the quality 1-10. "
            f"If quality >= 7, say '{APPROVED_VERDICT}'. "
            f"If quality < 7, give brief revision feedback, then say '{REVISION_VERDICT}'. "
            "Always end with exactly: SCORE: [number]"
        ),
        model=ModelConfig(id=model, temperature=0.3, max_tokens=256),
        max_iterations=2,
    )
    editor = Agent(profile=editor_profile, llm_provider=provider)

    # ── 3. Build the graph: writer always → editor; editor routes onward ────
    workflow = GraphWorkflow(name="write-review-loop", entry_point="writer")
    workflow.register_agent(writer)
    workflow.register_agent(editor)
    workflow.add_edge("writer", "editor")
    workflow.add_conditional_edge("editor", route_after_editor)

    # ── 4. Seed the state and run ────────────────────────────────────────────
    state = State(max_steps=MAX_STEPS).add_message(
        Message(role=MessageRole.USER, content=f"Write about: {TOPIC}")
    )
    print("Graph workflow: writer ↔ editor loop")
    print(f"Topic: {TOPIC}\n")
    print(DIVIDER)

    final_state = await workflow.run(state)
    print_conversation(final_state)

    print(DIVIDER)
    print(f"Status: {final_state.status.value} | Steps: {final_state.step_count}")


if __name__ == "__main__":
    asyncio.run(main())
