"""API key extraction and validation.

Reads X-API-Key header, resolves to a tier (anonymous/basic/premium).
Key definitions loaded from config/api_keys.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from fastapi import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

TIERS = ("anonymous", "basic", "premium")

# Rate limit config per tier
TIER_LIMITS: dict[str, dict] = {
    "anonymous": {"requests_per_minute": 30, "max_depth": 10, "complexity_budget": 500},
    "basic": {"requests_per_minute": 120, "max_depth": 15, "complexity_budget": 2000},
    "premium": {"requests_per_minute": 600, "max_depth": 20, "complexity_budget": 10000},
}


@dataclass(frozen=True)
class APIKeyInfo:
    tier: str
    key: str | None
    label: str | None = None


def _load_keys(config_path: Path) -> dict[str, APIKeyInfo]:
    """Load API keys from YAML config."""
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    keys = {}
    for entry in data.get("keys", []):
        key_val = entry.get("key", "")
        tier = entry.get("tier", "anonymous")
        if tier not in TIERS:
            tier = "anonymous"
        keys[key_val] = APIKeyInfo(tier=tier, key=key_val, label=entry.get("label"))
    return keys


def resolve_api_key(request: Request, config_path: Path | None = None) -> APIKeyInfo:
    """Resolve the API key from request headers.

    Args:
        request: The incoming HTTP request.
        config_path: Path to api_keys.yaml. Defaults to config/api_keys.yaml.

    Returns:
        APIKeyInfo with resolved tier.
    """
    if config_path is None:
        base = Path(__file__).resolve().parents[3]
        config_path = base / "config" / "api_keys.yaml"

    key_header = request.headers.get("x-api-key", "")
    if not key_header:
        return APIKeyInfo(tier="anonymous", key=None)

    keys = _load_keys(config_path)
    if key_header in keys:
        return keys[key_header]

    return APIKeyInfo(tier="anonymous", key=None)
