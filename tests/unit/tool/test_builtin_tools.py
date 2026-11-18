"""Behaviour tests for the built-in tools: what each returns and how it fails."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from miniagent.core.exceptions import ToolInvocationError
from miniagent.tool import (
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
    builtin_tools,
)


def _serve(monkeypatch: pytest.MonkeyPatch, handler: httpx.MockTransport) -> None:
    """Route every ``httpx.AsyncClient`` the tools create to ``handler``."""
    real_client = httpx.AsyncClient

    def client(**options: object) -> httpx.AsyncClient:
        return real_client(transport=handler, **options)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", client)


class TestCatalog:
    def test_should_offer_the_ten_documented_tools(self) -> None:
        assert sorted(builtin_tools()) == [
            "calculator",
            "echo",
            "file_reader",
            "json_formatter",
            "text_transform",
            "timestamp",
            "uuid_generator",
            "weather",
            "web_fetch",
            "word_count",
        ]


class TestCalculator:
    async def test_should_evaluate_arithmetic(self) -> None:
        assert await CalculatorTool().invoke(expression="(2 + 3) * 4") == "20.0"
        assert await CalculatorTool().invoke(expression="-7 // 2 + 2 ** 3 % 5") == "-1.0"

    async def test_should_refuse_anything_but_arithmetic(self) -> None:
        with pytest.raises(ToolInvocationError, match="Unsafe expression"):
            await CalculatorTool().invoke(expression="__import__('os').getcwd()")

    async def test_should_report_division_by_zero(self) -> None:
        with pytest.raises(ToolInvocationError, match="Division by zero"):
            await CalculatorTool().invoke(expression="1 / 0")

    async def test_should_reject_an_empty_or_malformed_expression(self) -> None:
        with pytest.raises(ToolInvocationError, match="empty"):
            await CalculatorTool().invoke(expression="  ")
        with pytest.raises(ToolInvocationError, match="Failed to evaluate"):
            await CalculatorTool().invoke(expression="2 +")


class TestFileReader:
    async def test_should_read_a_file_under_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        assert await FileReaderTool().invoke(path="notes.txt") == "hello"

    async def test_should_truncate_long_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "long.txt").write_text("x" * 50, encoding="utf-8")
        content = await FileReaderTool().invoke(path="long.txt", max_chars=10)
        assert content == "x" * 10 + "\n... [truncated at 10 characters]"

    async def test_should_deny_paths_outside_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
        monkeypatch.chdir(workspace)
        with pytest.raises(ToolInvocationError, match="outside the working directory"):
            await FileReaderTool().invoke(path="../secret.txt")

    async def test_should_deny_a_symlink_that_leaves_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
        (workspace / "link.txt").symlink_to(tmp_path / "secret.txt")
        monkeypatch.chdir(workspace)
        with pytest.raises(ToolInvocationError, match="outside the working directory"):
            await FileReaderTool().invoke(path="link.txt")

    async def test_should_report_missing_files_and_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "folder").mkdir()
        with pytest.raises(ToolInvocationError, match="File not found"):
            await FileReaderTool().invoke(path="missing.txt")
        with pytest.raises(ToolInvocationError, match="not a file"):
            await FileReaderTool().invoke(path="folder")
        with pytest.raises(ToolInvocationError, match="empty"):
            await FileReaderTool().invoke(path="")


class TestJsonFormatter:
    async def test_should_pretty_print_json(self) -> None:
        formatted = await JsonFormatterTool().invoke(json_string='{"a":[1,2]}', indent=2)
        assert formatted == json.dumps({"a": [1, 2]}, indent=2)

    async def test_should_report_invalid_json(self) -> None:
        with pytest.raises(ToolInvocationError, match="Invalid JSON"):
            await JsonFormatterTool().invoke(json_string="{not json")


class TestTextTransform:
    @pytest.mark.parametrize(
        ("transform", "expected"),
        [
            ("upper", " HELLO WORLD "),
            ("lower", " hello world "),
            ("title", " Hello World "),
            ("reverse", " dlroW olleh "),
            ("strip", "hello World"),
        ],
    )
    async def test_should_apply_each_transform(self, transform: str, expected: str) -> None:
        result = await TextTransformTool().invoke(text=" hello World ", transform=transform)
        assert result == expected

    async def test_should_reject_an_unknown_transform(self) -> None:
        with pytest.raises(ToolInvocationError, match="Unknown transform 'rot13'"):
            await TextTransformTool().invoke(text="x", transform="rot13")


class TestTimestamp:
    async def test_should_return_the_current_utc_time_in_iso_format(self) -> None:
        moment = datetime.fromisoformat(await TimestampTool().invoke())
        assert moment.utcoffset() is not None
        assert moment.utcoffset().total_seconds() == 0  # type: ignore[union-attr]

    async def test_should_apply_a_strftime_format(self) -> None:
        year = await TimestampTool().invoke(format="%Y")
        assert year.isdigit() and len(year) == 4


class TestUuid:
    @pytest.mark.parametrize("version", [1, 4])
    async def test_should_generate_the_requested_version(self, version: int) -> None:
        assert uuid.UUID(await UuidTool().invoke(version=version)).version == version

    async def test_should_default_to_version_four(self) -> None:
        assert uuid.UUID(await UuidTool().invoke()).version == 4

    async def test_should_reject_other_versions(self) -> None:
        with pytest.raises(ToolInvocationError, match="Unsupported UUID version: 3"):
            await UuidTool().invoke(version=3)


class TestWordCount:
    async def test_should_count_words_characters_and_lines(self) -> None:
        summary = await WordCountTool().invoke(text="one two\nthree")
        assert summary == "words: 3, characters: 13, lines: 2"

    async def test_should_count_empty_text_as_zero(self) -> None:
        assert await WordCountTool().invoke(text="") == "words: 0, characters: 0, lines: 0"


class TestEcho:
    async def test_should_return_the_message(self) -> None:
        assert await EchoTool().invoke(message="hello") == "hello"


class TestWebFetch:
    async def test_should_return_the_response_body_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _serve(monkeypatch, httpx.MockTransport(lambda request: httpx.Response(200, text="abcdef")))
        body = await WebFetchTool().invoke(url="https://example.com", max_chars=3)
        assert body == "abc\n... [truncated at 3 characters]"

    async def test_should_report_http_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve(monkeypatch, httpx.MockTransport(lambda request: httpx.Response(404)))
        with pytest.raises(ToolInvocationError, match="HTTP 404"):
            await WebFetchTool().invoke(url="https://example.com/missing")

    async def test_should_report_network_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        _serve(monkeypatch, httpx.MockTransport(refuse))
        with pytest.raises(ToolInvocationError, match="Failed to fetch"):
            await WebFetchTool().invoke(url="https://example.com")

    async def test_should_reject_an_empty_url(self) -> None:
        with pytest.raises(ToolInvocationError, match="URL cannot be empty"):
            await WebFetchTool().invoke(url=" ")


class TestWeather:
    async def test_should_ask_wttr_for_the_location(self, monkeypatch: pytest.MonkeyPatch) -> None:
        requested: list[httpx.URL] = []

        def reply(request: httpx.Request) -> httpx.Response:
            requested.append(request.url)
            return httpx.Response(200, text="Tokyo: +21°C\n")

        _serve(monkeypatch, httpx.MockTransport(reply))

        assert await WeatherTool().invoke(location="Tokyo") == "Tokyo: +21°C"
        assert requested[0].host == "wttr.in"
        assert requested[0].path == "/Tokyo"

    async def test_should_keep_a_location_from_changing_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requested: list[httpx.URL] = []

        def reply(request: httpx.Request) -> httpx.Response:
            requested.append(request.url)
            return httpx.Response(200, text="ok")

        _serve(monkeypatch, httpx.MockTransport(reply))

        await WeatherTool().invoke(location="New York/../?format=j1&x=#")

        assert requested[0].host == "wttr.in"
        assert requested[0].query == b"format=3"
        assert requested[0].raw_path.count(b"/") == 1

    async def test_should_report_service_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve(monkeypatch, httpx.MockTransport(lambda request: httpx.Response(503)))
        with pytest.raises(ToolInvocationError, match="HTTP 503"):
            await WeatherTool().invoke(location="Tokyo")

    async def test_should_reject_an_empty_location(self) -> None:
        with pytest.raises(ToolInvocationError, match="Location cannot be empty"):
            await WeatherTool().invoke(location="")
