"""Health-check and meta routes for the miniAgent API."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from miniagent import __version__

router = APIRouter(tags=["Meta"])


class HealthResponse(BaseModel):
    """Health-check response.

    Attributes:
        status: Always ``"ok"`` when the service is healthy.
        version: The running miniAgent version string.
    """

    status: str
    version: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service health status and version.",
)
def health() -> HealthResponse:
    """Return service health status.

    Returns:
        Health response with status ``"ok"`` and current version.
    """
    return HealthResponse(status="ok", version=__version__)


@router.get(
    "/",
    include_in_schema=False,
)
def root() -> RedirectResponse:
    """Redirect root requests to the OpenAPI docs UI.

    Returns:
        Redirect to ``/docs``.
    """
    return RedirectResponse(url="/docs")
