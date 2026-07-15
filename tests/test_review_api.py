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


# ---- Phase 3: review queue (event verdicts) ----


async def test_new_events_default_to_unreviewed(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/events/stilla-000001")
    assert r.json()["review"]["state"] == "unreviewed"


async def test_confirm_event(settings, run_id, client):
    r = await client.post(
        f"/api/runs/{run_id}/events/stilla-000001/review",
        data={"state": "confirmed", "reviewer": "Anna"},
    )
    assert r.status_code == 200
    assert r.json()["state"] == "confirmed"
    # The read paths reflect it.
    r = await client.get(f"/api/runs/{run_id}/events/stilla-000001")
    review = r.json()["review"]
    assert review["state"] == "confirmed"
    assert review["reviewer"] == "Anna"
    assert review["reviewed_at"] is not None
    r = await client.get(f"/api/runs/{run_id}/events")
    ev = next(e for e in r.json()["events"] if e["event_id"] == "stilla-000001")
    assert ev["review"]["state"] == "confirmed"


async def test_reject_event(settings, run_id, client):
    r = await client.post(f"/api/runs/{run_id}/events/hazard-000001/review", data={"state": "rejected"})
    assert r.status_code == 200
    assert r.json()["state"] == "rejected"


async def test_note_only_update_keeps_prior_state(settings, run_id, client):
    await client.post(f"/api/runs/{run_id}/events/stilla-000001/review", data={"state": "confirmed"})
    r = await client.post(
        f"/api/runs/{run_id}/events/stilla-000001/review",
        data={"note": "ser ut som en figurant"},
    )
    assert r.json()["state"] == "confirmed"
    assert r.json()["note"] == "ser ut som en figurant"


async def test_review_unknown_event_404(settings, run_id, client):
    r = await client.post(f"/api/runs/{run_id}/events/does-not-exist/review", data={"state": "confirmed"})
    assert r.status_code == 404


# ---- Phase 4: retroactive hazard marker ----


@pytest.fixture
def moving_run_id(settings: ReviewSettings) -> str:
    """A run with one tracklet moving steadily in +x — no MOT_FARA in the
    engine's own P5 output (no hazard was ever detected), so any MOT_FARA
    seen after placing a marker must come from the recompute path."""
    store = ArtifactStore(str(settings.output_dir), "vh-moving", "ch-moving")
    store.create()
    store.set_video_filename("film.mp4")
    store.record_pass_start(OfflineOrchestrator.P1_PASS_NAME, {"fps": 10.0})
    store.record_pass_complete(OfflineOrchestrator.P1_PASS_NAME, {})
    store.record_pass_start(OfflineOrchestrator.P2_PASS_NAME, {})
    store.start_fresh_pass_output("tracklets", OfflineOrchestrator.P2_PASS_NAME)
    for i in range(80):
        x = 50.0 + i * 4.0
        store.add_tracklet_frame(
            OfflineOrchestrator.P2_PASS_NAME,
            tracklet_id=1,
            frame_no=i,
            det_id=i,
            data={"cls": "person", "conf": 0.9, "xyxy": [x, 100.0, x + 30.0, 180.0]},
        )
    store.record_pass_complete(OfflineOrchestrator.P2_PASS_NAME, {"total_tracklet_rows": 80})
    store.record_pass_start(OfflineOrchestrator.P5_PASS_NAME, {"config": {}})
    store.start_fresh_pass_output("events", OfflineOrchestrator.P5_PASS_NAME)
    store.record_pass_complete(OfflineOrchestrator.P5_PASS_NAME, {"events_out": 0, "by_category": {}})
    store.close()
    return store.run_id


async def test_hazard_marker_inactive_by_default(settings, moving_run_id, client):
    r = await client.get(f"/api/runs/{moving_run_id}/hazard-marker")
    assert r.status_code == 200
    assert r.json() == {"active": False}


async def test_placing_hazard_marker_recomputes_mot_fara(settings, moving_run_id, client):
    # No MOT_FARA before the marker is placed (P5 emitted zero events).
    r = await client.get(f"/api/runs/{moving_run_id}/events")
    assert r.json()["count"] == 0

    r = await client.post(
        f"/api/runs/{moving_run_id}/hazard-marker",
        data={"x": "2000", "y": "140"},
    )
    assert r.status_code == 201
    assert r.json()["x"] == 2000.0

    r = await client.get(f"/api/runs/{moving_run_id}/hazard-marker")
    assert r.json() == {"active": True, "x": 2000.0, "y": 140.0, "note": None}

    r = await client.get(f"/api/runs/{moving_run_id}/events")
    body = r.json()
    assert body["count"] >= 1
    assert all(e["category"] == "MOT_FARA" for e in body["events"])


async def test_clearing_hazard_marker_reverts_to_engine_output(settings, moving_run_id, client):
    await client.post(f"/api/runs/{moving_run_id}/hazard-marker", data={"x": "2000", "y": "140"})
    r = await client.get(f"/api/runs/{moving_run_id}/events")
    assert r.json()["count"] >= 1

    r = await client.delete(f"/api/runs/{moving_run_id}/hazard-marker")
    assert r.status_code == 200
    assert r.json() == {"active": False}

    r = await client.get(f"/api/runs/{moving_run_id}/hazard-marker")
    assert r.json() == {"active": False}
    r = await client.get(f"/api/runs/{moving_run_id}/events")
    assert r.json()["count"] == 0


async def test_moving_marker_twice_to_same_position_is_deterministic(settings, moving_run_id, client):
    async def events_snapshot():
        r = await client.get(f"/api/runs/{moving_run_id}/events")
        return r.json()["events"]

    await client.post(f"/api/runs/{moving_run_id}/hazard-marker", data={"x": "2000", "y": "140"})
    first = await events_snapshot()
    await client.post(f"/api/runs/{moving_run_id}/hazard-marker", data={"x": "-500", "y": "400"})
    await client.post(f"/api/runs/{moving_run_id}/hazard-marker", data={"x": "2000", "y": "140"})
    back = await events_snapshot()
    assert first == back


@pytest.fixture
def two_person_moving_run_id(settings: ReviewSettings) -> str:
    """A run with two tracklets/persons: person 1 moves in -x (from x=3000
    down), person 2 moves in +x (from x=50 up). The engine's own P5 output
    already has MOT_FARA events for both (as if an earlier hazard was
    detected at x=2000, which both are approaching), spanning t=4.5-7.9 —
    the same span `recompute_mot_fara` derives for a marker placed at the
    same spot, so the reviewer's original verdict is checkable against the
    recompute. Moving the marker to x=3500 (beyond person 1's range, so
    person 1 is now moving away from it) makes only person 2 continue to
    qualify, and — since seq is assigned in tracklet-id order over however
    many tracklets qualify — person 2's recomputed event then gets a
    different event_id (seq 0 instead of 1) despite an identical time span
    and person_id: exactly the case where a verdict keyed by event_id alone
    would get lost."""
    store = ArtifactStore(str(settings.output_dir), "vh-two-moving", "ch-two-moving")
    store.create()
    store.set_video_filename("film.mp4")
    store.record_pass_start(OfflineOrchestrator.P1_PASS_NAME, {"fps": 10.0})
    store.record_pass_complete(OfflineOrchestrator.P1_PASS_NAME, {})
    store.record_pass_start(OfflineOrchestrator.P2_PASS_NAME, {})
    store.start_fresh_pass_output("tracklets", OfflineOrchestrator.P2_PASS_NAME)
    for i in range(80):
        x1 = 3000.0 - i * 4.0
        store.add_tracklet_frame(
            OfflineOrchestrator.P2_PASS_NAME,
            tracklet_id=1,
            frame_no=i,
            det_id=i,
            data={"cls": "person", "conf": 0.9, "xyxy": [x1, 100.0, x1 + 30.0, 180.0]},
        )
        x2 = 50.0 + i * 4.0
        store.add_tracklet_frame(
            OfflineOrchestrator.P2_PASS_NAME,
            tracklet_id=2,
            frame_no=i,
            det_id=1000 + i,
            data={"cls": "person", "conf": 0.9, "xyxy": [x2, 100.0, x2 + 30.0, 180.0]},
        )
    store.record_pass_complete(OfflineOrchestrator.P2_PASS_NAME, {"total_tracklet_rows": 160})
    store.record_pass_start(OfflineOrchestrator.P3_PASS_NAME, {})
    store.start_fresh_pass_output("persons", OfflineOrchestrator.P3_PASS_NAME)
    for tid, pid in ((1, 1), (2, 2)):
        store.add_person(
            OfflineOrchestrator.P3_PASS_NAME,
            pid,
            {
                "tracklet_ids": [tid],
                "embedding_centroids": {},
                "embedding_counts": {},
                "first_seen": 0.0,
                "last_seen": 7.9,
                "confirmation_state": "confirmed",
                "assoc_audit": [],
            },
        )
    store.record_pass_complete(OfflineOrchestrator.P3_PASS_NAME, {"persons_out": 2})
    store.record_pass_start(OfflineOrchestrator.P5_PASS_NAME, {"config": {}})
    store.start_fresh_pass_output("events", OfflineOrchestrator.P5_PASS_NAME)
    for eid, pid in (("mot_fara-000000", 1), ("mot_fara-000001", 2)):
        store.add_event(
            OfflineOrchestrator.P5_PASS_NAME,
            eid,
            {
                "category": "MOT_FARA",
                "person_id": pid,
                "t_start": 4.5,
                "t_end": 7.9,
                "confidence": 0.8,
                "evidence": {"tracklet_id": pid},
                "review": {"state": "unreviewed"},
            },
        )
    store.record_pass_complete(
        OfflineOrchestrator.P5_PASS_NAME, {"events_out": 2, "by_category": {"MOT_FARA": 2}}
    )
    store.close()
    return store.run_id


async def test_verdict_survives_hazard_marker_move_to_different_event_id(
    settings, two_person_moving_run_id, client
):
    # Confirm person 2's engine-original MOT_FARA event, before any marker
    # is placed.
    r = await client.post(
        f"/api/runs/{two_person_moving_run_id}/events/mot_fara-000001/review",
        data={"state": "confirmed", "reviewer": "Anna"},
    )
    assert r.status_code == 200

    # Marker A: both person 1 and person 2 qualify for MOT_FARA. Person 2's
    # event lands second (tracklet id order), keeping event_id mot_fara-000001.
    await client.post(f"/api/runs/{two_person_moving_run_id}/hazard-marker", data={"x": "2000", "y": "140"})
    r = await client.get(f"/api/runs/{two_person_moving_run_id}/events")
    events_a = r.json()["events"]
    assert len(events_a) == 2
    person2_event_a = next(e for e in events_a if e["person_id"] == 2)
    assert person2_event_a["review"]["state"] == "confirmed"
    assert person2_event_a["review"]["reviewer"] == "Anna"

    # Marker B: only person 2 still qualifies, so its recomputed event gets
    # a different event_id (seq 0 instead of 1) — but the same time span and
    # person_id. The prior verdict must still show up.
    await client.post(f"/api/runs/{two_person_moving_run_id}/hazard-marker", data={"x": "3500", "y": "140"})
    r = await client.get(f"/api/runs/{two_person_moving_run_id}/events")
    events_b = r.json()["events"]
    assert len(events_b) == 1
    person2_event_b = events_b[0]
    assert person2_event_b["person_id"] == 2
    assert person2_event_b["event_id"] != person2_event_a["event_id"]
    assert person2_event_b["review"]["state"] == "confirmed"
    assert person2_event_b["review"]["reviewer"] == "Anna"


async def test_review_invalid_state_422(settings, run_id, client):
    r = await client.post(f"/api/runs/{run_id}/events/stilla-000001/review", data={"state": "maybe"})
    assert r.status_code == 422


async def test_review_409_when_p5_not_run(settings, client):
    store = ArtifactStore(str(settings.output_dir), "vh", "ch")
    store.create()
    store.close()
    r = await client.post(f"/api/runs/{store.run_id}/events/ev-1/review", data={"state": "confirmed"})
    assert r.status_code == 409


# ---- Phase 3: operator-notes import ----


async def test_import_operator_notes(settings, run_id, client):
    text = "2 personer vid fordonet, 0:01\nrök vid ladan, 0:00"
    r = await client.post(f"/api/runs/{run_id}/operator-notes/import", data={"text": text})
    assert r.status_code == 201
    body = r.json()
    assert len(body["imported"]) == 2
    assert body["warnings"] == []
    r = await client.get(f"/api/runs/{run_id}/operator-notes")
    assert r.json()["count"] == 2


async def test_import_operator_notes_reports_warnings_but_keeps_good_lines(settings, run_id, client):
    text = "2 personer vid fordonet, 0:01\nden här raden går inte att tolka"
    r = await client.post(f"/api/runs/{run_id}/operator-notes/import", data={"text": text})
    assert r.status_code == 201
    body = r.json()
    assert len(body["imported"]) == 1
    assert len(body["warnings"]) == 1
    assert body["warnings"][0]["line"] == 2


async def test_delete_operator_note(settings, run_id, client):
    r = await client.post(f"/api/runs/{run_id}/operator-notes/import", data={"text": "text, 0:01"})
    aid = r.json()["imported"][0]["annotation_id"]
    r = await client.delete(f"/api/runs/{run_id}/operator-notes/{aid}")
    assert r.status_code == 200
    r = await client.get(f"/api/runs/{run_id}/operator-notes")
    assert r.json()["count"] == 0


async def test_delete_unknown_operator_note_404(settings, run_id, client):
    r = await client.delete(f"/api/runs/{run_id}/operator-notes/nonexistent")
    assert r.status_code == 404


# ---- Phase 3: comparison + debrief ----


async def test_comparison_buckets(settings, run_id, client):
    # stilla-000001 is at t_start=0.4; a note at t=0.5 is within tolerance.
    await client.post(f"/api/runs/{run_id}/operator-notes/import", data={"text": "text, 0:00.5"})
    r = await client.get(f"/api/runs/{run_id}/comparison")
    assert r.status_code == 200
    body = r.json()
    assert body["counts"]["both"] == 1
    assert body["counts"]["ai_only"] == 1  # hazard-000001 unmatched
    assert body["counts"]["operator_only"] == 0
    assert "delta_s" in body["both"][0]


async def test_comparison_tolerance_override(settings, run_id, client):
    await client.post(f"/api/runs/{run_id}/operator-notes/import", data={"text": "text, 5:00"})
    r = await client.get(f"/api/runs/{run_id}/comparison", params={"tolerance_s": 1})
    body = r.json()
    assert body["counts"]["both"] == 0
    assert body["counts"]["operator_only"] == 1


async def test_comparison_409_when_p5_not_run(settings, client):
    store = ArtifactStore(str(settings.output_dir), "vh", "ch")
    store.create()
    store.close()
    r = await client.get(f"/api/runs/{store.run_id}/comparison")
    assert r.status_code == 409


async def test_comparison_reflects_confirmed_verdict(settings, run_id, client):
    await client.post(f"/api/runs/{run_id}/events/stilla-000001/review", data={"state": "confirmed"})
    r = await client.get(f"/api/runs/{run_id}/comparison")
    ev = next(e for e in r.json()["ai_only"] if e["event_id"] == "stilla-000001")
    assert ev["review"]["state"] == "confirmed"


async def test_debrief_is_standalone_html(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/debrief")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "attachment" in r.headers["content-disposition"]
    assert "Hittad av båda" in r.text
    assert "<!doctype html>" in r.text.lower()


async def test_debrief_404_for_unknown_run(settings, client):
    r = await client.get("/api/runs/nonexistent/debrief")
    assert r.status_code == 404


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


@pytest.mark.asyncio
async def test_import_operator_notes_rejects_oversized_text(settings, run_id, client):
    text = "x" * 200_001
    r = await client.post(f"/api/runs/{run_id}/operator-notes/import", data={"text": text})
    assert r.status_code == 422
