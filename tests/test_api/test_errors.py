"""Tests for structured error responses."""

from __future__ import annotations

import pytest


class TestGraphQLErrors:
    def test_syntax_error(self, graphql):
        result = graphql("{ invalid syntax here }")
        assert "errors" in result

    def test_unknown_field(self, graphql):
        result = graphql("{ nonExistentField }")
        assert "errors" in result

    def test_valid_query_no_errors(self, graphql):
        result = graphql("{ health }")
        assert "errors" not in result

    def test_health_check_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
