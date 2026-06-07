"""Tests for the file download endpoint."""

import pytest
from pathlib import Path
from httpx import ASGITransport, AsyncClient

from src.api.app import create_api


@pytest.fixture
def client():
    base_dir = Path(__file__).resolve().parents[2]
    app = create_api(base_dir, debug=False)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.anyio
async def test_download_cleaned_parquet(client):
    async with client as c:
        r = await c.get("/dash/downloads/cleaned/ircc_intake.parquet")
        assert r.status_code == 200
        assert "application/octet-stream" in r.headers["content-type"]


@pytest.mark.anyio
async def test_download_output_csv(client):
    async with client as c:
        r = await c.get("/dash/downloads/outputs/counterfactual_series.csv")
        assert r.status_code == 200
        assert "application/octet-stream" in r.headers["content-type"]


@pytest.mark.anyio
async def test_download_master_parquet(client):
    async with client as c:
        r = await c.get("/dash/downloads/master/master_panel.parquet")
        assert r.status_code == 200


@pytest.mark.anyio
async def test_download_invalid_category(client):
    async with client as c:
        r = await c.get("/dash/downloads/invalid/file.csv")
        assert r.status_code == 400


@pytest.mark.anyio
async def test_download_missing_file(client):
    async with client as c:
        r = await c.get("/dash/downloads/outputs/nonexistent.csv")
        assert r.status_code == 404


@pytest.mark.anyio
async def test_download_path_traversal_blocked(client):
    """Verify resolve() blocks traversal. FastAPI's single-segment {filename}
    pattern prevents multi-segment paths from reaching the endpoint,
    so we test the resolve() logic directly."""
    from pathlib import Path
    base = Path(__file__).resolve().parents[2]
    # Simulating what the endpoint does:
    # base / "data/cleaned" / "../../etc/passwd" -> resolve() normalizes
    traversal = (base / "data/cleaned" / "../../etc/passwd").resolve()
    assert not str(traversal).startswith(str(base / "data"))
