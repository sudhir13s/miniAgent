"""Built-in agent profile templates and system prompts.

Pass a template to ``AgentProfile`` to get a ready-made agent type;
any field you set explicitly overrides the template.

Quick import::

    from miniagent.profile.templates import WEB_RESEARCHER

    profile = AgentProfile(
        name="market-scout",
        template=WEB_RESEARCHER,
        model=ModelConfig(id="openai/gpt-4o-mini"),
    )

Define your own template with ``ProfileTemplate``::

    from miniagent.profile.templates import ProfileTemplate

    SUPPORT_AGENT = ProfileTemplate(
        name="support_agent",
        system_prompt="You are a friendly customer-support agent.",
        tools=("web_fetch",),
    )
"""

from miniagent.profile.template import ProfileTemplate
from miniagent.profile.templates.catalog import (
    BUILTIN_TEMPLATES,
    CODE_REVIEWER,
    DOCUMENT_SUMMARIZER,
    GENERAL_ASSISTANT,
    MATH_SOLVER,
    PYTHON_DEVELOPER,
    TASK_PLANNER,
    TECHNICAL_WRITER,
    TOOL_ASSISTANT,
    WEB_RESEARCHER,
)
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

__all__ = [
    "BUILTIN_TEMPLATES",
    "CODE_REVIEWER",
    "CODE_REVIEWER_PROMPT",
    "DOCUMENT_SUMMARIZER",
    "DOCUMENT_SUMMARIZER_PROMPT",
    "GENERAL_ASSISTANT",
    "GENERAL_ASSISTANT_PROMPT",
    "MATH_SOLVER",
    "MATH_SOLVER_PROMPT",
    "PYTHON_DEVELOPER",
    "PYTHON_DEVELOPER_PROMPT",
    "TASK_PLANNER",
    "TASK_PLANNER_PROMPT",
    "TECHNICAL_WRITER",
    "TECHNICAL_WRITER_PROMPT",
    "TOOL_ASSISTANT",
    "TOOL_ASSISTANT_PROMPT",
    "WEB_RESEARCHER",
    "WEB_RESEARCHER_PROMPT",
    "ProfileTemplate",
]
