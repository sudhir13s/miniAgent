"""Test doubles for tool and toolset tests."""

from __future__ import annotations

from typing import Any

from miniagent.tool.base import BaseTool, ToolSchema
from miniagent.tool.toolset import BaseToolset


class StaticTool(BaseTool):
    """Tool that returns a fixed reply naming itself."""

    def __init__(self, name: str) -> None:
        super().__init__(ToolSchema(name=name, description=f"The {name} tool.", parameters={}))

    async def invoke(self, **kwargs: Any) -> str:
        return f"{self.name} ran"


class FakeToolset(BaseToolset):
    """Toolset whose tool list can change between calls, recording its lifecycle."""

    def __init__(self, name: str, tool_names: list[str], journal: list[str] | None = None) -> None:
        super().__init__(name)
        self.tool_names = tool_names
        self.events: list[str] = []
        self._journal = journal if journal is not None else []

    async def connect(self) -> None:
        self.events.append("connect")
        self._journal.append(f"connect {self.name}")

    async def close(self) -> None:
        self.events.append("close")
        self._journal.append(f"close {self.name}")

    async def list_tools(self) -> list[BaseTool]:
        return [StaticTool(tool_name) for tool_name in self.tool_names]
