"""Shared fixtures for API tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_api


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def base_dir():
    return PROJECT_ROOT


@pytest.fixture
def app(base_dir):
    return create_api(base_dir, debug=True)


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def graphql(client):
    """Helper to POST a GraphQL query and return the JSON response."""
    def _graphql(query: str, headers: dict | None = None):
        resp = client.post(
            "/graphql",
            json={"query": query},
            headers=headers or {},
        )
        return resp.json()
    return _graphql


@pytest.fixture
def basic_key():
    return "immath_basic_dev001"


@pytest.fixture
def premium_key():
    return "immath_premium_dev001"
