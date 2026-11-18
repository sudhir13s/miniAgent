"""The ReAct loop shared by ``Agent.run()`` and ``Agent.stream()``.

One implementation drives both, so a streamed run and a plain run always
behave the same: each iteration asks the model for a turn, runs any tools it
requested, and stops at the first accepted text reply or at
``max_iterations``. With an ``output_type``, a text reply is accepted only
once it validates (see ``structured_output``). The only difference between
the two entry points is whether the model turn is streamed.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from miniagent.agent.events import AgentEvent, RunCompleted, ToolCallFinished, ToolCallStarted
from miniagent.agent.structured_output import OUTPUT_VARIABLE, correction_message, parse_output
from miniagent.agent.tool_invocation import invoke_tool_calls
from miniagent.core.exceptions import AgentExecutionError
from miniagent.core.state import WorkflowStatus
from miniagent.llm.streaming import CompletionDone, StreamEvent, TextDelta

if TYPE_CHECKING:
    from miniagent.core.message import Message
    from miniagent.core.state import State
    from miniagent.llm.base import BaseLLMProvider
    from miniagent.profile.model_config import ModelConfig
    from miniagent.tool.base import BaseTool, ToolSchema

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoopSettings:
    """What one run of the loop needs from its agent.

    Attributes:
        agent_name: Used in log and error messages.
        max_iterations: Most model turns before the run stops.
        model_config: The model every turn uses.
        provider: The gateway to the model.
        tools: The tools available for this run, by name.
        fit_context: Picks the messages each model call receives.
        streaming: Whether model turns are streamed as ``TextDelta`` events.
        output_type: When set, the final reply must validate as this type.
    """

    agent_name: str
    max_iterations: int
    model_config: ModelConfig
    provider: BaseLLMProvider
    tools: Mapping[str, BaseTool]
    fit_context: Callable[[list[Message]], list[Message]]
    streaming: bool
    output_type: type[BaseModel] | None = None


async def run_react_loop(
    settings: LoopSettings, state: State, messages: list[Message]
) -> AsyncIterator[AgentEvent]:
    """Drive the loop, yielding events and finishing with ``RunCompleted``.

    Args:
        settings: The agent's model, tools and limits for this run.
        state: The run's starting state; new messages are added to it.
        messages: What the model sees: the history with the system prompt.

    Yields:
        Text deltas (when streaming), tool events, then one ``RunCompleted``.
    """
    progress = _Progress(state, messages, owns_status=state.status == WorkflowStatus.PENDING)
    tool_schemas = [tool.schema for tool in settings.tools.values()]
    for iteration in range(settings.max_iterations):
        turn: CompletionDone | None = None
        async for event in _model_turn(settings, settings.fit_context(messages), tool_schemas):
            if isinstance(event, TextDelta):
                yield event
            else:
                turn = event
        if turn is None:
            raise AgentExecutionError(
                f"Agent '{settings.agent_name}': the provider ended a turn without a reply."
            )
        progress.record(turn.message)
        if turn.tool_calls:
            async for tool_event in _tool_round(settings, turn, progress):
                yield tool_event
        elif _accept_final_reply(settings, progress, turn.message):
            logger.debug("Agent '%s' finished at iteration %d", settings.agent_name, iteration)
            yield progress.completed()
            return

    yield _out_of_iterations(settings, progress)


def _accept_final_reply(settings: LoopSettings, progress: _Progress, reply: Message) -> bool:
    """Accept a text reply, or ask for a correction when it fails ``output_type``.

    A valid reply's object is stored in ``state.variables["output"]``.
    """
    if settings.output_type is None:
        return True
    try:
        output = parse_output(reply.content, settings.output_type)
    except ValidationError as exc:
        logger.info(
            "Agent '%s' reply failed %s; asking again",
            settings.agent_name,
            settings.output_type.__name__,
        )
        progress.note(correction_message(exc))
        return False
    progress.state = progress.state.set_variable(OUTPUT_VARIABLE, output)
    return True


def _out_of_iterations(settings: LoopSettings, progress: _Progress) -> RunCompleted:
    """End a run that used every iteration: fail if structured output never arrived."""
    if settings.output_type is not None:
        raise AgentExecutionError(
            f"Agent '{settings.agent_name}' did not return a valid "
            f"{settings.output_type.__name__} within max_iterations={settings.max_iterations}."
        )
    logger.warning(
        "Agent '%s' hit max_iterations=%d without a final response.",
        settings.agent_name,
        settings.max_iterations,
    )
    return progress.completed()


@dataclass(slots=True)
class _Progress:
    """The run so far: the state to return and the messages the model sees.

    ``owns_status`` is true for a standalone run (the state arrived
    ``PENDING``); a workflow sets ``RUNNING`` first and keeps ownership.
    """

    state: State
    messages: list[Message]
    owns_status: bool

    def completed(self) -> RunCompleted:
        """End the run, marking a standalone run ``COMPLETED``."""
        if self.owns_status:
            self.state = self.state.model_copy(update={"status": WorkflowStatus.COMPLETED})
        return RunCompleted(self.state)

    def record(self, message: Message) -> None:
        """Add a message to both the model's view and the state, counting a step."""
        self.messages.append(message)
        self.state = self.state.add_message(message).increment_step()

    def note(self, message: Message) -> None:
        """Add a framework message (e.g. a correction) without counting a step."""
        self.messages.append(message)
        self.state = self.state.add_message(message)


async def _tool_round(
    settings: LoopSettings, turn: CompletionDone, progress: _Progress
) -> AsyncIterator[AgentEvent]:
    """Run the tools a turn requested and record their results in order."""
    for tool_call in turn.tool_calls:
        yield ToolCallStarted(tool_call)
    for result in await invoke_tool_calls(turn.tool_calls, settings.tools):
        progress.record(result)
        yield ToolCallFinished(result)


async def _model_turn(
    settings: LoopSettings, window: list[Message], tool_schemas: list[ToolSchema]
) -> AsyncIterator[StreamEvent]:
    """Get one model turn, streamed or not, offering tools only when there are any."""
    provider, model_config = settings.provider, settings.model_config
    if settings.streaming:
        async for event in provider.stream_completion(window, model_config, tool_schemas or None):
            yield event
        return
    if tool_schemas:
        message, tool_calls = await provider.complete_with_tools(window, model_config, tool_schemas)
    else:
        message, tool_calls = await provider.complete(window, model_config), []
    yield CompletionDone(message, tool_calls)
