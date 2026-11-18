"""Tools: the ``BaseTool`` contract, the ``@tool`` decorator, and the built-in tools.

Quick import::

    from miniagent.tool import BaseTool, CalculatorTool, ToolSchema, builtin_tools, tool
"""

from miniagent.tool.base import BaseTool, ToolSchema
from miniagent.tool.builtin import (
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
from miniagent.tool.catalog import builtin_tools
from miniagent.tool.function_tool import FunctionTool, tool
from miniagent.tool.toolset import BaseToolset

__all__ = [
    "BaseTool",
    "BaseToolset",
    "CalculatorTool",
    "EchoTool",
    "FileReaderTool",
    "FunctionTool",
    "JsonFormatterTool",
    "TextTransformTool",
    "TimestampTool",
    "ToolSchema",
    "UuidTool",
    "WeatherTool",
    "WebFetchTool",
    "WordCountTool",
    "builtin_tools",
    "tool",
]
