"""The one HTTP GET the built-in web tools share."""

from __future__ import annotations

import httpx

USER_AGENT = "miniAgent"


async def fetch_text(url: str, timeout_seconds: float) -> str:
    """Fetch a URL and return its response body as text, following redirects.

    Args:
        url: The address to fetch.
        timeout_seconds: Longest the request may take.

    Returns:
        The response body.

    Raises:
        httpx.HTTPStatusError: If the server answers with a 4xx or 5xx status.
        httpx.HTTPError: If the request fails or times out.
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_seconds) as client:
        response = await client.get(url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        return response.text
