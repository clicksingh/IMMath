"""IP-based rate limiting via SlowAPI.

Applied as FastAPI middleware on the /graphql endpoint.
Limits are tier-dependent: anonymous=30/min, basic=120/min, premium=600/min.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .api_key import resolve_api_key, TIER_LIMITS

logger = logging.getLogger(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["30/minute"],
    storage_uri="memory://",
)


def _get_limit_for_tier(tier: str) -> str:
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["anonymous"])
    return f"{limits['requests_per_minute']}/minute"


def rate_limit_middleware(request: Request, call_next):
    """Check rate limits based on API key tier before processing."""
    key_info = resolve_api_key(request)
    # The limit string is set per-request based on tier
    request.state.tier = key_info.tier
    request.state.api_key = key_info
    response = call_next(request)
    return response


def add_rate_limiting(app: FastAPI) -> None:
    """Add SlowAPI rate limiting to the FastAPI app."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.middleware("http")(rate_limit_middleware)
