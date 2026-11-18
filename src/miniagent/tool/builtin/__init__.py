"""Built-in tools — one tool per module plus the shared ``http_fetch`` helper.

No extra dependencies are required.
"""

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

__all__ = [
    "CalculatorTool",
    "EchoTool",
    "FileReaderTool",
    "JsonFormatterTool",
    "TextTransformTool",
    "TimestampTool",
    "UuidTool",
    "WeatherTool",
    "WebFetchTool",
    "WordCountTool",
]
