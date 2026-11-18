"""Turn a plain Python function into a tool with ``@tool``.

The tool's JSON schema comes from the function itself:

- type hints give each parameter's type (including Pydantic models, enums
  and ``Literal`` choices), and defaults make parameters optional;
- a Google-style docstring gives the tool description (its first paragraph)
  and each parameter's description (its ``Args:`` section).

Arguments from the model are validated against that schema before the
function runs, so the function receives correctly typed values::

    @tool
    async def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
        '''Get the current weather for a city.'''
        ...
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Callable
from typing import Any, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from miniagent.core.exceptions import ToolInvocationError
from miniagent.core.validation import describe_validation_error
from miniagent.tool.base import (
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    BaseTool,
    ToolSchema,
    validate_tool_name,
)

ToolFunction = Callable[..., Any]

_ARGS_SECTION_HEADERS = frozenset({"Args:", "Arguments:", "Parameters:"})
# "name: text" or "name (type): text" — the start of one parameter's entry.
_PARAMETER_ENTRY = re.compile(r"^\*{0,2}(?P<name>\w+)\s*(?:\([^)]*\))?\s*:\s*(?P<text>.*)$")
_SUPPORTED_PARAMETER_KINDS = frozenset(
    {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
)


class FunctionTool(BaseTool):
    """A tool backed by a Python function. Create one with ``@tool``.

    Sync functions run in a worker thread so they never block the event loop.
    A timed-out sync function cannot be interrupted; its thread finishes in
    the background while the agent moves on.
    """

    def __init__(
        self,
        function: ToolFunction,
        *,
        name: str | None = None,
        description: str | None = None,
        timeout_seconds: float | None = DEFAULT_TOOL_TIMEOUT_SECONDS,
    ) -> None:
        """Build the tool's schema from ``function``.

        Args:
            function: The sync or async function to expose.
            name: Tool name; defaults to the function's name.
            description: Tool description; defaults to the docstring's first paragraph.
            timeout_seconds: Per-call time limit; ``None`` disables it.

        Raises:
            TypeError: If a parameter lacks a type hint or is variadic or
                positional-only.
            ValueError: If the name is invalid or no description is available.
        """
        tool_name = validate_tool_name(name or function.__name__)
        summary, parameter_descriptions = _parse_docstring(inspect.getdoc(function))
        tool_description = description or summary
        if not tool_description:
            raise ValueError(
                f"Tool '{tool_name}' needs a description: add a docstring or pass description=."
            )
        self._function = function
        self._arguments_model = _build_arguments_model(function, tool_name, parameter_descriptions)
        super().__init__(
            ToolSchema(
                name=tool_name,
                description=tool_description,
                parameters=_parameters_schema(self._arguments_model),
            )
        )
        self.timeout_seconds = timeout_seconds

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function directly, without argument validation."""
        return self._function(*args, **kwargs)

    async def invoke(self, **kwargs: Any) -> str:
        """Validate the model's arguments, call the function, and return its result as text.

        Raises:
            ToolInvocationError: If the arguments do not match the schema.
        """
        arguments = self._validate(kwargs)
        if inspect.iscoroutinefunction(self._function):
            result = await self._function(**arguments)
        else:
            result = await asyncio.to_thread(self._function, **arguments)
        return _to_text(result)

    def _validate(self, raw_arguments: dict[str, Any]) -> dict[str, Any]:
        """Return the arguments converted to the declared parameter types."""
        try:
            validated = self._arguments_model.model_validate(raw_arguments)
        except ValidationError as exc:
            raise ToolInvocationError(
                f"Invalid arguments for tool '{self.name}': "
                f"{describe_validation_error(exc, root_label='arguments')}",
                cause=exc,
            ) from exc
        return {field: getattr(validated, field) for field in self._arguments_model.model_fields}


@overload
def tool(function: ToolFunction, /) -> FunctionTool: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    timeout_seconds: float | None = DEFAULT_TOOL_TIMEOUT_SECONDS,
) -> Callable[[ToolFunction], FunctionTool]: ...


def tool(
    function: ToolFunction | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout_seconds: float | None = DEFAULT_TOOL_TIMEOUT_SECONDS,
) -> FunctionTool | Callable[[ToolFunction], FunctionTool]:
    """Turn a function into a ``FunctionTool``; use as ``@tool`` or ``@tool(...)``.

    Args:
        function: The function, when used as a bare ``@tool``.
        name: Tool name; defaults to the function's name.
        description: Tool description; defaults to the docstring's first paragraph.
        timeout_seconds: Per-call time limit; ``None`` disables it.

    Returns:
        The tool, or a decorator that builds it.
    """

    def decorate(target: ToolFunction) -> FunctionTool:
        return FunctionTool(
            target, name=name, description=description, timeout_seconds=timeout_seconds
        )

    return decorate if function is None else decorate(function)


def _build_arguments_model(
    function: ToolFunction, tool_name: str, descriptions: dict[str, str]
) -> type[BaseModel]:
    """Build a Pydantic model whose fields mirror the function's parameters."""
    hints = get_type_hints(function, include_extras=True)
    fields: dict[str, Any] = {}
    for parameter in inspect.signature(function).parameters.values():
        if parameter.kind not in _SUPPORTED_PARAMETER_KINDS:
            raise TypeError(
                f"Tool '{tool_name}' cannot take *args, **kwargs or positional-only "
                f"parameters ('{parameter.name}'); declare each parameter by name."
            )
        if parameter.name not in hints:
            raise TypeError(f"Tool '{tool_name}' parameter '{parameter.name}' needs a type hint.")
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (
            hints[parameter.name],
            Field(default, description=descriptions.get(parameter.name)),
        )
    return create_model(
        f"{_pascal_case(tool_name)}Arguments",
        __config__=ConfigDict(extra="forbid", protected_namespaces=()),
        **fields,
    )


def _parameters_schema(arguments_model: type[BaseModel]) -> dict[str, Any]:
    """Return the model's JSON schema without the auto-generated titles the LLM doesn't need."""
    schema = arguments_model.model_json_schema()
    schema.pop("title", None)
    for property_schema in schema.get("properties", {}).values():
        property_schema.pop("title", None)
    return schema


def _parse_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into its summary and per-parameter descriptions."""
    if not docstring:
        return "", {}
    lines = docstring.splitlines()
    summary_lines: list[str] = []
    for line in lines:
        if not line.strip():
            break
        summary_lines.append(line.strip())
    return " ".join(summary_lines), _parse_args_section(lines)


def _parse_args_section(lines: list[str]) -> dict[str, str]:
    """Read ``name: description`` entries (with indented continuations) under ``Args:``."""
    descriptions: dict[str, str] = {}
    section_indent: int | None = None
    entry_indent = 0
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped in _ARGS_SECTION_HEADERS:
            section_indent, current = indent, None
            continue
        if section_indent is None:
            continue
        if not stripped or indent <= section_indent:
            section_indent = None
            continue
        entry = _PARAMETER_ENTRY.match(stripped)
        if entry and (current is None or indent <= entry_indent):
            name: str = entry["name"]
            current, entry_indent = name, indent
            descriptions[name] = entry["text"].strip()
        elif current is not None:
            descriptions[current] = f"{descriptions[current]} {stripped}".strip()
    return descriptions


def _to_text(result: Any) -> str:
    """Render a function's return value as the string a tool result must be."""
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return json.dumps(result, default=str)


def _pascal_case(name: str) -> str:
    """Convert ``snake_case`` or ``kebab-case`` to ``PascalCase``."""
    return "".join(part.capitalize() for part in re.split(r"[-_]", name))
