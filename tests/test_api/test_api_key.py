"""Tests for API key resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from src.api.middleware.api_key import resolve_api_key, APIKeyInfo, TIER_LIMITS


class TestAPIKeyResolution:
    def _make_request(self, headers_dict: dict[str, str]):
        from starlette.requests import Request
        from starlette.datastructures import Headers

        scope = {
            "type": "http",
            "headers": [
                (k.lower().encode(), v.encode()) for k, v in headers_dict.items()
            ],
        }
        return Request(scope)

    def test_no_key_is_anonymous(self):
        """Missing X-API-Key header defaults to anonymous tier."""
        request = self._make_request({})
        info = resolve_api_key(request)
        assert info.tier == "anonymous"
        assert info.key is None

    def test_basic_key(self):
        request = self._make_request({"x-api-key": "immath_basic_dev001"})
        info = resolve_api_key(request)
        assert info.tier == "basic"
        assert info.key == "immath_basic_dev001"

    def test_premium_key(self):
        request = self._make_request({"x-api-key": "immath_premium_dev001"})
        info = resolve_api_key(request)
        assert info.tier == "premium"

    def test_invalid_key_falls_to_anonymous(self):
        request = self._make_request({"x-api-key": "invalid_key_12345"})
        info = resolve_api_key(request)
        assert info.tier == "anonymous"

    def test_key_is_case_sensitive(self):
        request = self._make_request({"x-api-key": "IMMATH_BASIC_DEV001"})
        info = resolve_api_key(request)
        assert info.tier == "anonymous"  # uppercase not in config


class TestAPIKeyWithGraphQL:
    def test_query_with_basic_key(self, client):
        resp = client.post(
            "/graphql",
            json={"query": "{ health }"},
            headers={"X-API-Key": "immath_basic_dev001"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["health"] == "ok"

    def test_query_with_premium_key(self, client):
        resp = client.post(
            "/graphql",
            json={"query": "{ health }"},
            headers={"X-API-Key": "immath_premium_dev001"},
        )
        assert resp.status_code == 200
