"""FastAPI application factory for miniAgent.

Creates and configures the FastAPI application, registering all routers,
middleware, and startup behaviour.

Usage::

    # Run directly
    python -m miniagent

    # Or import the factory for testing
    from miniagent.api.app import create_app
    app = create_app()
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from miniagent import __version__
from miniagent.api.dependencies import create_session_store
from miniagent.api.lifespan import lifespan
from miniagent.api.routes.agents import router as agent_router
from miniagent.api.routes.meta import router as meta_router
from miniagent.api.routes.workflows import router as workflow_router
from miniagent.api.settings import ServerSettings

logger = logging.getLogger(__name__)

WILDCARD_ORIGIN = "*"
# The API only exposes GET and POST routes.
CORS_METHODS = ["GET", "POST"]

# Type alias for the request-logging middleware's ``call_next`` argument.
_CallNext = Callable[[Request], Awaitable[Response]]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Cross-origin browser access is off unless ``CORS_ORIGINS`` lists the
    allowed origins (comma-separated, or ``*`` for any origin).

    Returns:
        A fully configured ``FastAPI`` instance.
    """
    app = FastAPI(
        title="miniAgent API",
        description=(
            "REST interface for the miniAgent framework. "
            "Run agents and orchestrate multi-agent workflows."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    # Replaced by the lifespan with the connected MCP toolsets.
    app.state.toolsets = []
    app.state.session_store = create_session_store()

    _add_cors(app, list(ServerSettings.from_environment().cors_origins))

    # ── Request logging middleware ────────────────────────────────────────────
    @app.middleware("http")
    async def _log_requests(request: Request, call_next: _CallNext) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(meta_router)
    app.include_router(agent_router)
    app.include_router(workflow_router)

    return app


def _add_cors(app: FastAPI, origins: list[str]) -> None:
    """Let browsers on ``origins`` call the API; do nothing when none are listed.

    Credentials are never allowed: the API has no cookie or session auth,
    and browsers reject a wildcard origin combined with credentials.
    """
    if not origins:
        return
    if WILDCARD_ORIGIN in origins:
        logger.warning("CORS allows every origin; list explicit origins in production.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=CORS_METHODS,
        allow_headers=["*"],
    )
