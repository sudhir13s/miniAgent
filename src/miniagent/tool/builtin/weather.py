"""WeatherTool — returns current weather for a city using wttr.in (no API key needed)."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from miniagent.core.exceptions import ToolInvocationError
from miniagent.tool.base import BaseTool, ToolSchema
from miniagent.tool.builtin.http_fetch import fetch_text

# wttr.in's one-line format: "<location>: <condition> <temperature>".
_WEATHER_URL = "https://wttr.in/{location}?format=3"
_REQUEST_TIMEOUT_SECONDS = 10.0


class WeatherTool(BaseTool):
    """Returns current weather for a city using wttr.in (no API key needed).

    Uses the free wttr.in service which requires no registration or API key.

    Example::

        tool = WeatherTool()
        result = await tool.invoke(location="London")
        # "London: ⛅️ +15°C"
    """

    def __init__(self) -> None:
        """Initialise the weather tool with its JSON schema."""
        super().__init__(
            ToolSchema(
                name="weather",
                description=(
                    "Get the current weather for a city or location. "
                    "Uses the free wttr.in service — no API key required. "
                    "Example locations: 'London', 'New York', 'Tokyo', 'Paris'."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": (
                                "City name or location, e.g. 'London' or 'San Francisco'."
                            ),
                        },
                    },
                    "required": ["location"],
                },
            )
        )

    async def invoke(self, **kwargs: Any) -> str:
        """Fetch and return the current weather for ``location`` via wttr.in."""
        location: str = kwargs.get("location", "").strip()
        if not location:
            raise ToolInvocationError("Location cannot be empty.")

        # Quoted so a location cannot add its own path or query to the request.
        url = _WEATHER_URL.format(location=quote(location, safe=""))
        try:
            return (await fetch_text(url, _REQUEST_TIMEOUT_SECONDS)).strip()
        except httpx.HTTPStatusError as exc:
            raise ToolInvocationError(
                f"Weather service returned HTTP {exc.response.status_code} for '{location}'.",
                cause=exc,
            ) from exc
        except Exception as exc:
            raise ToolInvocationError(
                f"Failed to fetch weather for '{location}': {exc}", cause=exc
            ) from exc
