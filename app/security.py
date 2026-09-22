"""
Shared-secret authentication between the Java backend and this service.

The bridge is an internal microservice: it is meant to be reachable only by the
backend, never by browsers. When an API key is configured every call must carry
it; when none is configured the service stays open, which is convenient while
developing on one machine but must not be used on a shared network.
"""

from __future__ import annotations

import hmac
import logging
from typing import Optional

from fastapi import Header, HTTPException, status

from .config import settings

LOGGER = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias=API_KEY_HEADER)) -> None:
    """FastAPI dependency that rejects calls without the shared secret."""
    if not settings.auth_enabled:
        return

    # Compared as bytes so that a key with non-ASCII characters cannot break the check.
    provided = (x_api_key or "").encode("utf-8")
    expected = settings.api_key.encode("utf-8")

    if not provided or not hmac.compare_digest(provided, expected):
        LOGGER.warning("Rejected a request with a missing or invalid API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid API key is required.",
        )
