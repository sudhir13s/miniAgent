"""Agent — the miniAgent ReAct agent.

Implements the full ReAct (Reason + Act) loop:

    1. Build messages from current state; an optional ``ContextPolicy``
       picks which of them the model receives.
    2. Call ``BaseLLMProvider.complete_with_tools()`` (``complete()`` if no
       tools, ``stream_completion()`` when streaming).
    3. Append the AI message to state.
    4. If tool calls are present, invoke them concurrently and append one
       ``TOOL`` message per call, in the order the model requested them.
    5. Repeat until no tool calls are requested or ``max_iterations`` is reached.

``run()`` returns the final state; ``stream()`` yields the same run as events
(text deltas, tool calls, completion). Both share ``react_loop``.

A failing tool never aborts the run: unknown tools, invalid arguments,
exceptions and timeouts all come back to the model as error results so it can
correct itself (see ``miniagent.agent.tool_invocation``). Provider failures
still raise ``AgentExecutionError``.

The agent talks to models only through a ``BaseLLMProvider``. The default
``LiteLLMProvider`` reaches any model LiteLLM supports, selected by the
profile's model ID (e.g. ``"openai/gpt-4o"``, ``"gemini/gemini-2.0-flash"``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from miniagent.agent.base import BaseAgent
from miniagent.agent.events import AgentEvent, RunCompleted
from miniagent.agent.react_loop import LoopSettings, run_react_loop
from miniagent.agent.structured_output import output_instructions
from miniagent.core.exceptions import AgentExecutionError
from miniagent.core.message import Message, MessageRole

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

    from miniagent.core.state import State
    from miniagent.llm.base import BaseLLMProvider
    from miniagent.memory.context_window import ContextPolicy
    from miniagent.profile.agent_profile import AgentProfile
    from miniagent.tool.base import BaseTool
    from miniagent.tool.toolset import BaseToolset

logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """The miniAgent ``BaseAgent`` implementation: a ReAct loop over an LLM provider.

    The agent runs a ReAct loop capped at ``profile.max_iterations`` iterations.
    On each iteration it either produces a final text response (loop ends) or
    generates tool calls which are executed before the next iteration.

    Attributes:
        profile: Declarative agent configuration (model, system prompt, tools).
        llm_provider: The LLM gateway used for completions.
        tools: Mapping of tool name → ``BaseTool`` instance, fixed at construction.
        toolsets: Tool sources whose tools are discovered at the start of each run.
        context_policy: Chooses the messages sent on each model call; ``None``
            sends the full history. The returned state always keeps it all.
        output_type: Pydantic model the final answer must validate as, or
            ``None`` for a free-text answer.
    """

    def __init__(
        self,
        profile: AgentProfile,
        llm_provider: BaseLLMProvider,
        tools: dict[str, BaseTool] | None = None,
        *,
        toolsets: Sequence[BaseToolset] | None = None,
        context_policy: ContextPolicy | None = None,
        output_type: type[BaseModel] | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            profile: The agent's declarative configuration.
            llm_provider: The LLM gateway to use for completions.
            tools: Optional mapping of tool name → tool instance.
            toolsets: Optional tool sources (e.g. MCP servers) resolved per run.
            context_policy: Optional policy choosing which messages each model
                call receives; without one the full history is sent.
            output_type: Optional Pydantic model the final answer must match;
                the validated object is stored in ``state.variables["output"]``.
        """
        super().__init__(profile)
        self.llm_provider = llm_provider
        self.tools: dict[str, BaseTool] = tools or {}
        self.toolsets: list[BaseToolset] = list(toolsets or [])
        self.context_policy = context_policy
        self.output_type = output_type

    async def run(self, state: State) -> State:
        """Run the ReAct loop and return the updated state.

        Builds the message history from ``state``, adds the system prompt,
        then iterates until a text-only response is produced or
        ``max_iterations`` is exhausted.

        Args:
            state: The current workflow state.

        Returns:
            Updated state with new messages appended.

        Raises:
            AgentExecutionError: If the LLM provider or a toolset fails, two
                tools share a name, the state's step limit is exceeded, or
                no reply validates as ``output_type`` within
                ``max_iterations``.
        """
        async for event in self._events(state, streaming=False):
            if isinstance(event, RunCompleted):
                return event.state
        raise AgentExecutionError(f"Agent '{self.profile.name}' stopped without completing.")

    async def stream(self, state: State) -> AsyncIterator[AgentEvent]:
        """Run the ReAct loop, yielding events as the reply is written.

        Behaves exactly like ``run()`` but streams each model turn: text
        arrives as ``TextDelta`` events, tools as ``ToolCallStarted`` and
        ``ToolCallFinished``, and the last event is ``RunCompleted`` with
        the same final state ``run()`` would return.

        Args:
            state: The current workflow state.

        Yields:
            ``AgentEvent`` objects, ending with one ``RunCompleted``.

        Raises:
            AgentExecutionError: On the same failures as ``run()``.
        """
        async for event in self._events(state, streaming=True):
            yield event

    def _with_system_prompt(self, history: Sequence[Message]) -> list[Message]:
        """Return a copy of ``history`` that starts with the profile's system prompt.

        A system message already present in the history wins, so callers can
        override the profile's prompt for a single run. With an
        ``output_type``, the first system message also gets the JSON Schema
        the answer must match.
        """
        messages = list(history)
        if not any(message.role == MessageRole.SYSTEM for message in messages):
            system = Message(role=MessageRole.SYSTEM, content=self.profile.system_prompt)
            messages.insert(0, system)
        if self.output_type is None:
            return messages
        index = next(i for i, message in enumerate(messages) if message.role == MessageRole.SYSTEM)
        system = messages[index]
        instructions = output_instructions(self.output_type)
        messages[index] = system.model_copy(
            update={"content": f"{system.content}\n\n{instructions}"}
        )
        return messages

    def _context_window(self, messages: list[Message]) -> list[Message]:
        """Return a snapshot of the messages the next model call receives.

        A copy, so the provider never sees the agent's later appends.
        """
        if self.context_policy is None:
            return list(messages)
        return self.context_policy.fit(messages, self._count_tokens)

    def _count_tokens(self, messages: Sequence[Message]) -> int:
        """Count tokens for this agent's model, as the provider measures them."""
        return self.llm_provider.count_tokens(messages, self.profile.model)

    async def _events(self, state: State, *, streaming: bool) -> AsyncIterator[AgentEvent]:
        """Run the shared loop, turning any failure into ``AgentExecutionError``."""
        messages = self._with_system_prompt(state.messages)
        try:
            settings = LoopSettings(
                agent_name=self.profile.name,
                max_iterations=self.profile.max_iterations,
                model_config=self.profile.model,
                provider=self.llm_provider,
                tools=await self._resolve_tools(),
                fit_context=self._context_window,
                streaming=streaming,
                output_type=self.output_type,
            )
            logger.debug(
                "Agent '%s' starting run: messages=%d, tools=%d, streaming=%s",
                self.profile.name,
                len(messages),
                len(settings.tools),
                streaming,
            )
            async for event in run_react_loop(settings, state, messages):
                yield event
        except AgentExecutionError:
            raise
        except Exception as exc:
            raise AgentExecutionError(
                f"Agent '{self.profile.name}' failed: {exc}", cause=exc
            ) from exc

    async def _resolve_tools(self) -> dict[str, BaseTool]:
        """Combine the agent's fixed tools with what each toolset offers right now.

        Raises:
            AgentExecutionError: If two sources provide a tool with the same name,
                since the model could not tell them apart.
        """
        tools = dict(self.tools)
        sources = dict.fromkeys(tools, "the agent's own tools")
        listings = await asyncio.gather(*(toolset.list_tools() for toolset in self.toolsets))
        for toolset, toolset_tools in zip(self.toolsets, listings, strict=True):
            for tool in toolset_tools:
                if tool.name in tools:
                    raise AgentExecutionError(
                        f"Agent '{self.profile.name}' has two tools named '{tool.name}': "
                        f"from {sources[tool.name]} and from toolset '{toolset.name}'."
                    )
                tools[tool.name] = tool
                sources[tool.name] = f"toolset '{toolset.name}'"
        return tools
