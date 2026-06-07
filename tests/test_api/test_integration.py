"""End-to-end integration tests — GraphQL + Dash coexistence."""

from __future__ import annotations

import pytest


class TestHealthCheck:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestGraphQLPost:
    def test_graphql_endpoint(self, client):
        resp = client.post("/graphql", json={"query": "{ health }"})
        assert resp.status_code == 200
        assert resp.json()["data"]["health"] == "ok"


class TestDashCoexistence:
    def test_root_redirects_to_dash(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (200, 307, 308)

    def test_dash_loads(self, client):
        resp = client.get("/dash/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_dash_title(self, client):
        resp = client.get("/dash/")
        assert "ACI Research Tool" in resp.text


class TestCORSHeaders:
    def test_cors_on_graphql(self, client):
        resp = client.options(
            "/graphql",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        # CORS should allow the origin
        assert resp.status_code in (200, 204)


class TestFullRoundTrip:
    def test_query_all_models(self, graphql):
        """Verify all 6 data models return valid data in one query."""
        result = graphql("""
        {
            health
            cohortNiv(first: 1) { totalCount }
            lambdaResults { variable }
            welfareLoss(first: 1) { totalCount }
            decomposition(first: 1) { totalCount }
            counterfactual(first: 1) { totalCount }
            masterPanel(first: 1) { totalCount }
        }
        """)
        data = result["data"]
        assert data["health"] == "ok"
        assert data["cohortNiv"]["totalCount"] > 0
        assert len(data["lambdaResults"]) > 0
        assert data["welfareLoss"]["totalCount"] > 0
        assert data["decomposition"]["totalCount"] > 0
        assert data["counterfactual"]["totalCount"] > 0
        assert data["masterPanel"]["totalCount"] > 0
