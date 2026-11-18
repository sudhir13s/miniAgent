"""Unit tests for the HTTP GET the built-in web tools share."""

from __future__ import annotations

import httpx
import pytest

from miniagent.tool.builtin.http_fetch import USER_AGENT, fetch_text


def _serve(monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport) -> None:
    real_client = httpx.AsyncClient

    def client(**options: object) -> httpx.AsyncClient:
        return real_client(transport=transport, **options)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", client)


class TestFetchText:
    async def test_should_return_the_body_and_identify_itself(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[httpx.Request] = []

        def reply(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, text="hello")

        _serve(monkeypatch, httpx.MockTransport(reply))

        assert await fetch_text("https://example.com/page", timeout_seconds=5) == "hello"
        assert seen[0].headers["user-agent"] == USER_AGENT

    async def test_should_follow_redirects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def reply(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/old":
                return httpx.Response(302, headers={"Location": "https://example.com/new"})
            return httpx.Response(200, text="moved here")

        _serve(monkeypatch, httpx.MockTransport(reply))

        assert await fetch_text("https://example.com/old", timeout_seconds=5) == "moved here"

    async def test_should_raise_for_an_error_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _serve(monkeypatch, httpx.MockTransport(lambda request: httpx.Response(500)))

        with pytest.raises(httpx.HTTPStatusError):
            await fetch_text("https://example.com", timeout_seconds=5)
