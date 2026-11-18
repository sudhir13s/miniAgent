"""Name-based lookup for the built-in tools.

Profiles and YAML configs list tools by name (``tools: [calculator]``);
this module maps those names to fresh tool instances.
"""

from __future__ import annotations

from collections.abc import Callable

from miniagent.tool.base import BaseTool
from miniagent.tool.builtin.calculator import CalculatorTool
from miniagent.tool.builtin.echo import EchoTool
from miniagent.tool.builtin.file_reader import FileReaderTool
from miniagent.tool.builtin.json_formatter import JsonFormatterTool
from miniagent.tool.builtin.text_transform import TextTransformTool
from miniagent.tool.builtin.timestamp import TimestampTool
from miniagent.tool.builtin.uuid_generator import UuidTool
from miniagent.tool.builtin.weather import WeatherTool
from miniagent.tool.builtin.web_fetch import WebFetchTool
from miniagent.tool.builtin.word_count import WordCountTool

# Built-in tools take no constructor arguments; each class is a zero-arg factory.
_BUILTIN_TOOL_CLASSES: tuple[Callable[[], BaseTool], ...] = (
    CalculatorTool,
    EchoTool,
    FileReaderTool,
    JsonFormatterTool,
    TextTransformTool,
    TimestampTool,
    UuidTool,
    WeatherTool,
    WebFetchTool,
    WordCountTool,
)


def builtin_tools() -> dict[str, BaseTool]:
    """Create one fresh instance of every built-in tool, keyed by tool name.

    Returns:
        Mapping of tool name (as the LLM sees it) to tool instance.
    """
    instances = (tool_class() for tool_class in _BUILTIN_TOOL_CLASSES)
    return {tool.name: tool for tool in instances}
