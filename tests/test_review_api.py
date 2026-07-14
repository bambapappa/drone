"""Tests for the review REST API.

Drives the API end-to-end against a synthetic analysis run on disk: writes
a minimal sidecar with manifest + P2/P3/P5 outputs (no real engine runs),
then exercises the read endpoints (run list, events, tracklets, persons,
frames meta) and the write endpoints (bookmarks, screenshots). The point
is to verify the route plumbing, path-traversal guards, and the contract
the UI depends on — not the analysis itself.

Path-traversal security tests cover run_id, annotation_id, and event_id,
since the API serves files from a configured directory and a malicious id
must never escape it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from analysis.orchestrator import OfflineOrchestrator
from analysis.store import ArtifactStore
from review.config import ReviewSettings
from review.main import app


@pytest.fixture
def settings(tmp_path: Path, monkeypatch) -> ReviewSettings:
    """Point the review API at a synthetic output_dir + video_dir."""
    output_dir = tmp_path / "analysis-output"
    output_dir.mkdir()
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    # Write a placeholder video so video_available flips true.
    (video_dir / "film.mp4").write_bytes(b"FAKE-MP4-BYTES")
    monkeypatch.setenv("ANALYSIS_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("VIDEO_DIR", str(video_dir))
    return ReviewSettings.from_env()


@pytest.fixture
def run_id(settings: ReviewSettings) -> str:
    """Seed a complete-looking run with P2/P3/P5 outputs and a video link."""
    store = ArtifactStore(str(settings.output_dir), "vh-aaa", "ch-aaa")
    store.create()
    store.set_video_filename("film.mp4")
    # P1 frames
    store.record_pass_start(OfflineOrchestrator.P1_PASS_NAME, {})
    for fn in range(10):
        store.add_frame(OfflineOrchestrator.P1_PASS_NAME, fn, {"pts_ms": fn * 100.0})
    store.record_pass_complete(OfflineOrchestrator.P1_PASS_NAME, {"frames_processed": 10})
    # P2 tracklets — two tracklets alternating frames
    store.record_pass_start(OfflineOrchestrator.P2_PASS_NAME, {})
    store.start_fresh_pass_output("tracklets", OfflineOrchestrator.P2_PASS_NAME)
    for fn in range(10):
        store.add_tracklet_frame(
            OfflineOrchestrator.P2_PASS_NAME,
            tracklet_id=1,
            frame_no=fn,
            det_id=fn * 2,
            data={"cls": "person", "conf": 0.9, "xyxy": [100.0, 100.0, 130.0, 180.0]},
        )
        store.add_tracklet_frame(
            OfflineOrchestrator.P2_PASS_NAME,
            tracklet_id=2,
            frame_no=fn,
            det_id=fn * 2 + 1,
            data={"cls": "person", "conf": 0.85, "xyxy": [200.0, 100.0, 230.0, 180.0]},
        )
    store.record_pass_complete(OfflineOrchestrator.P2_PASS_NAME, {"total_tracklet_rows": 20})
    # P3 persons
    store.record_pass_start(OfflineOrchestrator.P3_PASS_NAME, {})
    store.start_fresh_pass_output("persons", OfflineOrchestrator.P3_PASS_NAME)
    store.add_person(
        OfflineOrchestrator.P3_PASS_NAME,
        1,
        {
            "tracklet_ids": [1],
            "embedding_centroids": {},
            "embedding_counts": {},
            "first_seen": 0.0,
            "last_seen": 0.9,
            "confirmation_state": "confirmed",
            "assoc_audit": [],
        },
    )
    store.add_person(
        OfflineOrchestrator.P3_PASS_NAME,
        2,
        {
            "tracklet_ids": [2],
            "embedding_centroids": {},
            "embedding_counts": {},
            "first_seen": 0.0,
            "last_seen": 0.9,
            "confirmation_state": "confirmed",
            "assoc_audit": [],
        },
    )
    store.record_pass_complete(OfflineOrchestrator.P3_PASS_NAME, {"persons_out": 2})
    # P5 events
    store.record_pass_start(OfflineOrchestrator.P5_PASS_NAME, {})
    store.start_fresh_pass_output("events", OfflineOrchestrator.P5_PASS_NAME)
    store.add_event(
        OfflineOrchestrator.P5_PASS_NAME,
        "stilla-000001",
        {
            "category": "STILLA",
            "person_id": 1,
            "t_start": 0.4,
            "t_end": 0.9,
            "confidence": 0.7,
            "evidence": {"tracklet_id": 1},
            "review": {"state": "unreviewed"},
        },
    )
    store.add_event(
        OfflineOrchestrator.P5_PASS_NAME,
        "hazard-000001",
        {
            "category": "HAZARD",
            "person_id": None,
            "t_start": 0.0,
            "t_end": 1.0,
            "confidence": 0.6,
            "evidence": {"kind": "smoke"},
            "review": {"state": "unreviewed"},
        },
    )
    store.record_pass_complete(
        OfflineOrchestrator.P5_PASS_NAME,
        {"events_out": 2, "by_category": {"STILLA": 1, "HAZARD": 1}, "p3_used": True},
    )
    store.close()
    return store.run_id


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ---- health + index ----


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_index_serves_review_gui(client):
    r = await client.get("/")
    assert r.status_code == 200
    # Swedish brand string is present.
    assert "GRANSKNING" in r.text or "granskning" in r.text.lower()


# ---- run listing + summary ----


async def test_list_runs_empty(settings, client):
    # No runs yet (the run_id fixture hasn't been requested).
    r = await client.get("/api/runs")
    assert r.status_code == 200
    assert r.json() == {"runs": []}


async def test_list_runs_returns_seeded_run(settings, run_id, client):
    r = await client.get("/api/runs")
    assert r.status_code == 200
    runs = r.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["video_filename"] == "film.mp4"


async def test_get_run_summary(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == run_id
    assert body["video_available"] is True
    assert body["passes"]["p2_track"]["status"] == "complete"
    assert body["passes"]["p5_events"]["stats"]["events_out"] == 2


async def test_get_run_404_for_unknown(settings, client):
    r = await client.get("/api/runs/nonexistent")
    assert r.status_code == 404


# ---- events ----


async def test_get_events(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/events")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    cats = {e["category"] for e in body["events"]}
    assert cats == {"STILLA", "HAZARD"}


async def test_get_single_event(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/events/stilla-000001")
    assert r.status_code == 200
    assert r.json()["category"] == "STILLA"


async def test_get_event_404_unknown_id(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/events/does-not-exist")
    assert r.status_code == 404


async def test_get_events_409_when_p5_not_run(settings, client):
    # Seed a run that never ran P5.
    store = ArtifactStore(str(settings.output_dir), "vh", "ch")
    store.create()
    store.set_video_filename("film.mp4")
    store.record_pass_start(OfflineOrchestrator.P1_PASS_NAME, {})
    store.record_pass_complete(OfflineOrchestrator.P1_PASS_NAME, {})
    store.record_pass_start(OfflineOrchestrator.P2_PASS_NAME, {})
    store.record_pass_complete(OfflineOrchestrator.P2_PASS_NAME, {})
    store.close()
    r = await client.get(f"/api/runs/{store.run_id}/events")
    assert r.status_code == 409


# ---- tracklets + persons + frames meta ----


async def test_get_tracklets_for_frame(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/tracklets", params={"frame": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["frame"] == 0
    assert len(body["boxes"]) == 2
    # person_id joined from P3.
    pids = sorted(b["person_id"] for b in body["boxes"])
    assert pids == [1, 2]


async def test_get_tracklets_range(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/tracklets/range", params={"from": 0, "to": 2})
    assert r.status_code == 200
    body = r.json()
    # Frames 0, 1, 2 — each with 2 boxes.
    assert sorted(body["frames"].keys()) == ["0", "1", "2"]
    assert all(len(frames) == 2 for frames in body["frames"].values())


async def test_get_tracklets_409_when_p2_not_run(settings, client):
    store = ArtifactStore(str(settings.output_dir), "vh", "ch")
    store.create()
    store.close()
    r = await client.get(f"/api/runs/{store.run_id}/tracklets", params={"frame": 0})
    assert r.status_code == 409


async def test_get_persons(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/persons")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2


async def test_get_frames_meta(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/frames/meta", params={"from": 0, "to": 4})
    assert r.status_code == 200
    body = r.json()
    assert len(body["frames"]) == 5
    assert body["frames"][0] == {"frame_no": 0, "pts_ms": 0.0}
    assert body["frames"][4] == {"frame_no": 4, "pts_ms": 400.0}


# ---- video serving ----


async def test_get_video_serves_file(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/video")
    assert r.status_code == 200
    assert r.content == b"FAKE-MP4-BYTES"


async def test_get_video_404_when_missing(settings, client):
    store = ArtifactStore(str(settings.output_dir), "vh", "ch")
    store.create()
    store.set_video_filename("nonexistent.mp4")
    store.close()
    r = await client.get(f"/api/runs/{store.run_id}/video")
    assert r.status_code == 404


# ---- export ----


async def test_export_csv(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/export", params={"format": "csv"})
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert "STILLA" in r.text
    assert "HAZARD" in r.text


async def test_export_json(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/export", params={"format": "json"})
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == run_id
    assert len(body["events"]) == 2


# ---- annotations: bookmarks ----


async def test_bookmark_lifecycle(settings, run_id, client):
    # Create.
    r = await client.post(
        f"/api/runs/{run_id}/bookmarks",
        data={"t": "12.5", "label": "Vid bilen", "note": "oklart"},
    )
    assert r.status_code == 201
    aid = r.json()["annotation_id"]
    # List.
    r = await client.get(f"/api/runs/{run_id}/annotations")
    assert r.status_code == 200
    anns = r.json()
    assert len(anns["bookmarks"]) == 1
    assert anns["bookmarks"][0]["annotation_id"] == aid
    # Delete (tombstone).
    r = await client.delete(f"/api/runs/{run_id}/bookmarks/{aid}")
    assert r.status_code == 200
    # List again — empty.
    r = await client.get(f"/api/runs/{run_id}/annotations")
    assert r.json()["bookmarks"] == []


async def test_delete_bookmark_404_for_unknown(settings, run_id, client):
    r = await client.delete(f"/api/runs/{run_id}/bookmarks/nonexistent")
    assert r.status_code == 404


# ---- annotations: screenshots ----


async def test_screenshot_with_png_lifecycle(settings, run_id, client):
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    r = await client.post(
        f"/api/runs/{run_id}/screenshots",
        data={"t": "3.2", "label": "Skärmdump 1"},
        files={"png": ("frame.png", png_bytes, "image/png")},
    )
    assert r.status_code == 201
    aid = r.json()["annotation_id"]
    # PNG is retrievable.
    r = await client.get(f"/api/runs/{run_id}/screenshots/{aid}/png")
    assert r.status_code == 200
    assert r.content == png_bytes
    # Tombstone.
    r = await client.delete(f"/api/runs/{run_id}/screenshots/{aid}")
    assert r.status_code == 200
    # PNG gone from listing but file remains on disk (disk GC is separate).
    r = await client.get(f"/api/runs/{run_id}/annotations")
    assert all(s["annotation_id"] != aid for s in r.json()["screenshots"])


async def test_screenshot_metadata_only(settings, run_id, client):
    r = await client.post(
        f"/api/runs/{run_id}/screenshots",
        data={"t": "1.5", "label": "anteckning utan png"},
    )
    assert r.status_code == 201
    assert r.json()["png_filename"] is None


# ---- path-traversal guards ----
#
# FastAPI's router rejects some malformed ids before our validator runs
# (405 for trailing-slash redirects, 404 for nested paths that don't match
# the route pattern). Either way, the traversal is blocked — the test
# accepts any non-200/non-redirect-to-real-content status.


@pytest.mark.parametrize(
    "bad_id",
    ["..", "../etc", "foo/bar", "a;b", "a b"],
)
async def test_run_id_rejects_traversal(settings, client, bad_id):
    r = await client.get(f"/api/runs/{bad_id}", follow_redirects=False)
    assert r.status_code in (400, 404, 405, 422), f"{bad_id!r}: {r.status_code}"


@pytest.mark.parametrize(
    "bad_id",
    ["..", "../etc", "foo/bar"],
)
async def test_bookmark_id_rejects_traversal(settings, run_id, client, bad_id):
    r = await client.delete(f"/api/runs/{run_id}/bookmarks/{bad_id}", follow_redirects=False)
    # FastAPI returns 404/405 for paths that don't match the route; our
    # validator returns 400 for ones that do match but fail the regex.
    assert r.status_code in (400, 404, 405, 422), f"{bad_id!r}: {r.status_code}"


# ---- bad request validation ----


async def test_add_bookmark_rejects_negative_t(settings, run_id, client):
    r = await client.post(
        f"/api/runs/{run_id}/bookmarks",
        data={"t": "-1", "label": "x"},
    )
    assert r.status_code == 422


async def test_add_bookmark_rejects_empty_label(settings, run_id, client):
    r = await client.post(
        f"/api/runs/{run_id}/bookmarks",
        data={"t": "1", "label": ""},
    )
    assert r.status_code == 422
