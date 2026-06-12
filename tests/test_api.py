"""API surface tests (no pipeline started — lifespan not triggered via transport)."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_index_serves_gui(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "INSATS" in r.text


async def test_state(client):
    r = await client.get("/api/state")
    assert r.status_code == 200
    body = r.json()
    assert "pipeline" in body and "config" in body


async def test_videos_list(client):
    r = await client.get("/api/videos")
    assert r.status_code == 200
    assert isinstance(r.json()["videos"], list)


async def test_source_rejects_traversal(client):
    r = await client.post("/api/source", json={"name": "../../etc/passwd"})
    assert r.status_code in (400, 404)


async def test_source_rejects_bad_ext(client):
    r = await client.post("/api/source", json={"name": "evil.sh"})
    assert r.status_code == 400


async def test_danger_requires_pipeline(client):
    r = await client.post("/api/danger", json={"x": 0.5, "y": 0.5})
    assert r.status_code == 409


async def test_clear_danger_idempotent(client):
    r = await client.delete("/api/danger")
    assert r.status_code == 200
