"""Workflows: the ``BaseWorkflow`` contract and sequential, parallel, loop, graph runners.

Quick import::

    from miniagent.workflow import GraphWorkflow, SequentialWorkflow, WorkflowRegistry
"""

from miniagent.workflow.base import BaseWorkflow
from miniagent.workflow.graph import END_NODE, GraphWorkflow
from miniagent.workflow.loop import LoopWorkflow
from miniagent.workflow.parallel import ParallelWorkflow
from miniagent.workflow.registry import WorkflowRegistry
from miniagent.workflow.sequential import SequentialWorkflow

__all__ = [
    "END_NODE",
    "BaseWorkflow",
    "GraphWorkflow",
    "LoopWorkflow",
    "ParallelWorkflow",
    "SequentialWorkflow",
    "WorkflowRegistry",
]
