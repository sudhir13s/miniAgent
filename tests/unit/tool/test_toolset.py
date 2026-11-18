"""Tests for the ``BaseToolset`` contract."""

from __future__ import annotations

import pytest

from miniagent.tool.base import BaseTool
from miniagent.tool.toolset import BaseToolset, open_toolsets
from tests.unit.tool.fakes import FakeToolset


class _MinimalToolset(BaseToolset):
    async def list_tools(self) -> list[BaseTool]:
        return []


class TestBaseToolset:
    async def test_context_manager_connects_and_closes(self) -> None:
        toolset = FakeToolset("docs", ["search"])

        async with toolset as entered:
            assert entered is toolset
            assert toolset.events == ["connect"]

        assert toolset.events == ["connect", "close"]

    async def test_context_manager_closes_when_body_raises(self) -> None:
        toolset = FakeToolset("docs", ["search"])

        with pytest.raises(RuntimeError):
            async with toolset:
                raise RuntimeError("boom")

        assert toolset.events == ["connect", "close"]

    async def test_connect_and_close_default_to_no_ops(self) -> None:
        async with _MinimalToolset("plain") as toolset:
            assert await toolset.list_tools() == []

    async def test_open_toolsets_connects_all_and_closes_in_reverse(self) -> None:
        journal: list[str] = []
        first = FakeToolset("first", [], journal)
        second = FakeToolset("second", [], journal)

        async with open_toolsets([first, second]) as opened:
            assert opened == [first, second]

        assert journal == ["connect first", "connect second", "close second", "close first"]

    async def test_open_toolsets_closes_opened_ones_when_a_later_connect_fails(self) -> None:
        journal: list[str] = []

        class _Unreachable(FakeToolset):
            async def connect(self) -> None:
                raise ConnectionError("server down")

        with pytest.raises(ConnectionError):
            async with open_toolsets([FakeToolset("first", [], journal), _Unreachable("down", [])]):
                pass

        assert journal == ["connect first", "close first"]

    def test_name_must_be_an_identifier(self) -> None:
        with pytest.raises(ValueError, match="Toolset name"):
            _MinimalToolset("has space")
