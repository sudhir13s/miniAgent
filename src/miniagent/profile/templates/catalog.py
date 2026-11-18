"""Built-in agent profile templates.

Every template is model-agnostic: combine it with any ``ModelConfig``
when building an ``AgentProfile``. Tool names refer to the built-in
tools in ``miniagent.tool``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from miniagent.profile.template import ProfileTemplate
from miniagent.profile.templates.prompts import (
    CODE_REVIEWER_PROMPT,
    DOCUMENT_SUMMARIZER_PROMPT,
    GENERAL_ASSISTANT_PROMPT,
    MATH_SOLVER_PROMPT,
    PYTHON_DEVELOPER_PROMPT,
    TASK_PLANNER_PROMPT,
    TECHNICAL_WRITER_PROMPT,
    TOOL_ASSISTANT_PROMPT,
    WEB_RESEARCHER_PROMPT,
)

GENERAL_ASSISTANT = ProfileTemplate(
    name="general-assistant",
    description="General-purpose Q&A assistant without tools.",
    system_prompt=GENERAL_ASSISTANT_PROMPT,
    max_iterations=5,
)

TOOL_ASSISTANT = ProfileTemplate(
    name="tool-assistant",
    description="Assistant that answers with arithmetic, time, and weather tools.",
    system_prompt=TOOL_ASSISTANT_PROMPT,
    tools=("calculator", "timestamp", "weather"),
    max_iterations=10,
)

WEB_RESEARCHER = ProfileTemplate(
    name="web-researcher",
    description="Researches a topic from web pages and local files, citing sources.",
    system_prompt=WEB_RESEARCHER_PROMPT,
    tools=("web_fetch", "file_reader"),
    max_iterations=10,
)

TECHNICAL_WRITER = ProfileTemplate(
    name="technical-writer",
    description="Turns source material into structured developer documentation.",
    system_prompt=TECHNICAL_WRITER_PROMPT,
    max_iterations=5,
)

PYTHON_DEVELOPER = ProfileTemplate(
    name="python-developer",
    description="Writes and modifies production-quality Python code.",
    system_prompt=PYTHON_DEVELOPER_PROMPT,
    tools=("file_reader",),
    max_iterations=15,
)

CODE_REVIEWER = ProfileTemplate(
    name="code-reviewer",
    description="Reviews source files for bugs, security issues, and maintainability.",
    system_prompt=CODE_REVIEWER_PROMPT,
    tools=("file_reader",),
    max_iterations=10,
)

MATH_SOLVER = ProfileTemplate(
    name="math-solver",
    description="Solves arithmetic word problems step by step with a calculator.",
    system_prompt=MATH_SOLVER_PROMPT,
    tools=("calculator",),
    max_iterations=10,
)

TASK_PLANNER = ProfileTemplate(
    name="task-planner",
    description="Breaks a goal into ordered, verifiable tasks with risks.",
    system_prompt=TASK_PLANNER_PROMPT,
    max_iterations=5,
)

DOCUMENT_SUMMARIZER = ProfileTemplate(
    name="document-summarizer",
    description="Summarizes a local document into key points and action items.",
    system_prompt=DOCUMENT_SUMMARIZER_PROMPT,
    tools=("file_reader", "word_count"),
    max_iterations=5,
)

BUILTIN_TEMPLATES: Mapping[str, ProfileTemplate] = MappingProxyType(
    {
        template.name: template
        for template in (
            GENERAL_ASSISTANT,
            TOOL_ASSISTANT,
            WEB_RESEARCHER,
            TECHNICAL_WRITER,
            PYTHON_DEVELOPER,
            CODE_REVIEWER,
            MATH_SOLVER,
            TASK_PLANNER,
            DOCUMENT_SUMMARIZER,
        )
    }
)
"""Read-only mapping of template name to built-in template."""
