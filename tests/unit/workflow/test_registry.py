"""Tests for WorkflowRegistry."""

from __future__ import annotations

import pytest

from miniagent.core.exceptions import WorkflowExecutionError
from miniagent.workflow.registry import WorkflowRegistry
from miniagent.workflow.sequential import SequentialWorkflow


class TestWorkflowRegistry:
    """Tests for WorkflowRegistry."""

    def _make_registry(self) -> WorkflowRegistry:
        return WorkflowRegistry()

    def test_register_and_get(self) -> None:
        wf = SequentialWorkflow("stub")
        reg = self._make_registry()
        reg.register("my-wf", wf)
        assert reg.get("my-wf") is wf

    def test_get_unknown_raises(self) -> None:
        reg = self._make_registry()
        with pytest.raises(WorkflowExecutionError, match="No workflow"):
            reg.get("nonexistent")

    def test_duplicate_registration_raises(self) -> None:
        wf = SequentialWorkflow("stub")
        reg = self._make_registry()
        reg.register("wf", wf)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("wf", wf)

    def test_unregister(self) -> None:
        wf = SequentialWorkflow("stub")
        reg = self._make_registry()
        reg.register("wf", wf)
        reg.unregister("wf")
        assert "wf" not in reg

    def test_unregister_unknown_raises(self) -> None:
        reg = self._make_registry()
        with pytest.raises(WorkflowExecutionError):
            reg.unregister("ghost")

    def test_list_workflows_sorted(self) -> None:
        reg = self._make_registry()
        reg.register("zzz", SequentialWorkflow("z"))
        reg.register("aaa", SequentialWorkflow("a"))
        assert reg.list_workflows() == ["aaa", "zzz"]

    def test_len(self) -> None:
        reg = self._make_registry()
        reg.register("a", SequentialWorkflow("a"))
        reg.register("b", SequentialWorkflow("b"))
        assert len(reg) == 2

    def test_contains(self) -> None:
        reg = self._make_registry()
        reg.register("wf", SequentialWorkflow("wf"))
        assert "wf" in reg
        assert "other" not in reg
