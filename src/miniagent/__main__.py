"""Entry point for ``python -m miniagent``: start the API server with uvicorn.

The server is configured by environment variables; see
``miniagent.api.settings`` for every variable and its default.
"""

from __future__ import annotations

import uvicorn

from miniagent.api.settings import ServerSettings

_APP_FACTORY = "miniagent.api.app:create_app"


def main() -> None:
    """Launch the miniAgent API server."""
    settings = ServerSettings.from_environment()
    uvicorn.run(
        _APP_FACTORY,
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=settings.reload,
    )


if __name__ == "__main__":
    main()
