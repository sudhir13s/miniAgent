"""WebFetchTool — fetches the text content of a URL."""

from __future__ import annotations

from typing import Any

import httpx

from miniagent.core.exceptions import ToolInvocationError
from miniagent.tool.base import BaseTool, ToolSchema
from miniagent.tool.builtin.http_fetch import fetch_text

DEFAULT_MAX_CHARS = 5000
_REQUEST_TIMEOUT_SECONDS = 15.0


class WebFetchTool(BaseTool):
    """Fetches the text content of a URL.

    Returns the raw response text (HTML, JSON, plain text, etc.).
    Large responses are truncated to ``max_chars`` characters.

    Example::

        tool = WebFetchTool()
        result = await tool.invoke(url="https://example.com", max_chars=2000)
        # "<!doctype html><html>..."
    """

    def __init__(self) -> None:
        """Initialise the web fetch tool with its JSON schema."""
        super().__init__(
            ToolSchema(
                name="web_fetch",
                description=(
                    "Fetch the text content of a URL (HTML, JSON, plain text, etc.). "
                    "Returns up to max_chars characters of the response body."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch, e.g. 'https://example.com'.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": (
                                f"Maximum characters to return (default: {DEFAULT_MAX_CHARS})."
                            ),
                            "default": DEFAULT_MAX_CHARS,
                        },
                    },
                    "required": ["url"],
                },
            )
        )

    async def invoke(self, **kwargs: Any) -> str:
        """Fetch ``url`` and return up to ``max_chars`` of the response body."""
        url: str = kwargs.get("url", "").strip()
        max_chars: int = int(kwargs.get("max_chars", DEFAULT_MAX_CHARS))
        if not url:
            raise ToolInvocationError("URL cannot be empty.")

        try:
            text = await fetch_text(url, _REQUEST_TIMEOUT_SECONDS)
        except httpx.HTTPStatusError as exc:
            raise ToolInvocationError(
                f"HTTP {exc.response.status_code} fetching '{url}'.", cause=exc
            ) from exc
        except Exception as exc:
            raise ToolInvocationError(f"Failed to fetch '{url}': {exc}", cause=exc) from exc
        if len(text) > max_chars:
            return text[:max_chars] + f"\n... [truncated at {max_chars} characters]"
        return text
