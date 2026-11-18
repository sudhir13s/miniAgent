"""Toolsets: sources of tools that are discovered at run time.

A plain ``BaseTool`` is fixed when the agent is built. A toolset instead
answers "which tools do you have right now?" each time an agent runs, which
fits tool servers such as MCP that publish their own tool list and may change
it. Toolsets that hold a connection open it in ``connect`` and release it in
``close``; use them as async context managers::

    github = MCPToolset.http("github", "https://example.com/mcp")
    async with github:
        agent = build_agent(profile, toolsets=[github])
        await agent.run(state)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, TypeVar

from miniagent.core.naming import validate_identifier

if TYPE_CHECKING:
    from types import TracebackType

    from miniagent.tool.base import BaseTool

_ToolsetT = TypeVar("_ToolsetT", bound="BaseToolset")


class BaseToolset(ABC):
    """A named source of tools, resolved at the start of every agent run.

    Attributes:
        name: Identifier profiles use to select this toolset (``toolsets: [name]``).
    """

    def __init__(self, name: str) -> None:
        """Initialise the toolset.

        Args:
            name: Identifier for the toolset; letters, digits, hyphens and underscores.

        Raises:
            ValueError: If the name contains any other character.
        """
        self.name = validate_identifier(name, "Toolset")

    @abstractmethod
    async def list_tools(self) -> list[BaseTool]:
        """Return the tools this source currently offers."""

    async def connect(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Open any connection the toolset needs. Does nothing by default."""

    async def close(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release resources opened by ``connect``. Does nothing by default."""

    async def __aenter__(self: _ToolsetT) -> _ToolsetT:
        """Connect and return the toolset."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the toolset, even when the body raised."""
        await self.close()


@asynccontextmanager
async def open_toolsets(toolsets: Iterable[BaseToolset]) -> AsyncIterator[list[BaseToolset]]:
    """Connect several toolsets for the duration of a block, then close them all.

    Toolsets close in reverse order, and every one that connected is closed
    even if a later one fails to connect or the block raises.

    Args:
        toolsets: The toolsets to open.

    Yields:
        The opened toolsets, in the order given.
    """
    opened = list(toolsets)
    async with AsyncExitStack() as exit_stack:
        for toolset in opened:
            await exit_stack.enter_async_context(toolset)
        yield opened
