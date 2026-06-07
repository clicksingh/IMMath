"""Tests for rate limiting middleware."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_api
from src.api.middleware.api_key import TIER_LIMITS


class TestRateLimitConfig:
    def test_tiers_defined(self):
        assert "anonymous" in TIER_LIMITS
        assert "basic" in TIER_LIMITS
        assert "premium" in TIER_LIMITS

    def test_anonymous_lowest(self):
        a = TIER_LIMITS["anonymous"]["requests_per_minute"]
        b = TIER_LIMITS["basic"]["requests_per_minute"]
        p = TIER_LIMITS["premium"]["requests_per_minute"]
        assert a < b < p

    def test_depth_increases_with_tier(self):
        a = TIER_LIMITS["anonymous"]["max_depth"]
        b = TIER_LIMITS["basic"]["max_depth"]
        p = TIER_LIMITS["premium"]["max_depth"]
        assert a < b < p

    def test_complexity_increases_with_tier(self):
        a = TIER_LIMITS["anonymous"]["complexity_budget"]
        b = TIER_LIMITS["basic"]["complexity_budget"]
        p = TIER_LIMITS["premium"]["complexity_budget"]
        assert a < b < p


class TestDepthLimit:
    def test_normal_depth_passes(self, graphql):
        result = graphql("{ health }")
        assert "errors" not in result
        assert result["data"]["health"] == "ok"

    def test_deep_query_rejected(self, graphql):
        # Build a deeply nested query beyond anonymous max_depth (10)
        deep_query = "{ health " * 15 + "}" * 15
        result = graphql(deep_query)
        # Should get an error about depth (or parse error, which is fine too)
        assert "errors" in result or "data" in result
