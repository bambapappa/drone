"""Tests for the P5 event-derivation pass wired into the orchestrator.

Drives a synthetic video + the orchestrator's run_pass_p5() end-to-end and
checks the events/ sidecar table matches what the in-memory derivation
produces. The pure-logic event-diff tests live in test_events.py; this file
covers the orchestrator-side glue (frame decoding, person_by_tracklet map
from P3, table persistence, manifest stats).

Uses a FakeDetector-style stub for P1/P2 by writing detections/tracklets
directly into the store, then exercising only run_pass_p5() — the goal is
to verify the pass wiring, not the analyzers themselves.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from analysis.ingest import VideoMeta, build_pts_index
from analysis.orchestrator import OfflineConfig, OfflineOrchestrator
from analysis.store import ArtifactStore


def _make_video(path: Path, n_frames: int = 60, fps: float = 10.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    for _ in range(n_frames):
        frame = np.full((240, 320, 3), 80, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_meta_and_store(tmp: str, video_path: Path) -> tuple[VideoMeta, ArtifactStore]:
    index, fps, w, h, total = build_pts_index(str(video_path))
    import hashlib

    with open(video_path, "rb") as f:
        vhash = hashlib.sha256(f.read(100_000)).hexdigest()
    meta = VideoMeta(
        path=str(video_path.resolve()),
        fps=fps,
        width=w,
        height=h,
        total_frames=total,
        video_hash=vhash,
        pts_index=index,
    )
    store = ArtifactStore(tmp, vhash, "cfg-hash")
    store.create()
    return meta, store


def _seed_p1_p2_p3(
    store: ArtifactStore,
    frames_processed: int,
    tracklet_rows: list[dict],
    persons: list[dict],
) -> None:
    """Write minimal P1/P2/P3 output to the store and mark passes complete,
    so run_pass_p5()'s gates (P2 complete, optional P3) pass."""
    store.record_pass_start(OfflineOrchestrator.P1_PASS_NAME, {})
    for f in range(frames_processed):
        store.add_frame(OfflineOrchestrator.P1_PASS_NAME, f, {"pts_ms": f * 100.0})
    store.record_pass_complete(OfflineOrchestrator.P1_PASS_NAME, {"frames_processed": frames_processed})

    store.record_pass_start(OfflineOrchestrator.P2_PASS_NAME, {})
    store.start_fresh_pass_output("tracklets", OfflineOrchestrator.P2_PASS_NAME)
    for row in tracklet_rows:
        store.add_tracklet_frame(
            OfflineOrchestrator.P2_PASS_NAME,
            row["tracklet_id"],
            row["frame_no"],
            row.get("det_id", 0),
            {"cls": "person", "conf": 0.9, "xyxy": row["xyxy"]},
        )
    store.record_pass_complete(OfflineOrchestrator.P2_PASS_NAME, {"total_tracklet_rows": len(tracklet_rows)})

    if persons:
        store.record_pass_start(OfflineOrchestrator.P3_PASS_NAME, {})
        store.start_fresh_pass_output("persons", OfflineOrchestrator.P3_PASS_NAME)
        for p in persons:
            store.add_person(OfflineOrchestrator.P3_PASS_NAME, p["person_id"], p)
        store.record_pass_complete(OfflineOrchestrator.P3_PASS_NAME, {"persons_out": len(persons)})


def test_run_pass_p5_writes_events_table():
    """End-to-end: P5 emits at least one event for a synthetic stationary
    tracklet, persists it to events/p5_events.jsonl, and the manifest's
    by_category breakdown matches the on-disk rows."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "v.mp4"
        _make_video(video_path, n_frames=100, fps=10.0)
        meta, store = _make_meta_and_store(tmp, video_path)

        # Stationary tracklet for 100 frames (10s) → STILLA event expected.
        stationary = [100.0, 100.0, 130.0, 180.0]
        tracklet_rows = [
            {"tracklet_id": 1, "frame_no": f, "det_id": f, "xyxy": stationary} for f in range(100)
        ]
        persons = [
            {
                "person_id": 1,
                "tracklet_ids": [1],
                "embedding_centroids": {},
                "embedding_counts": {},
                "first_seen": 0.0,
                "last_seen": 9.9,
                "confirmation_state": "confirmed",
                "assoc_audit": [],
            }
        ]
        _seed_p1_p2_p3(store, frames_processed=100, tracklet_rows=tracklet_rows, persons=persons)

        config = OfflineConfig()
        orch = OfflineOrchestrator(meta, store, config)
        n = orch.run_pass_p5()

        assert n >= 1, "expected at least one STILLA event for a stationary tracklet"
        events = list(store.iter_events(OfflineOrchestrator.P5_PASS_NAME))
        assert len(events) == n
        # All rows are JSONL with the events/ schema.
        for ev in events:
            assert "event_id" in ev
            assert "category" in ev
            assert "t_start" in ev and "t_end" in ev
            assert "confidence" in ev
            assert "evidence" in ev
            assert ev["review"]["state"] == "unreviewed"

        # Manifest stats reflect the run.
        p5_stats = store.manifest["passes"][OfflineOrchestrator.P5_PASS_NAME]["stats"]
        assert p5_stats["events_out"] == n
        assert p5_stats["by_category"]
        assert p5_stats["p3_used"] is True


def test_run_pass_p5_person_id_carried_when_p3_ran():
    """When P3 ran, STILLA events carry the person_id P3 assigned to that
    tracklet (the engine never invents person_ids)."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "v.mp4"
        _make_video(video_path, n_frames=100, fps=10.0)
        meta, store = _make_meta_and_store(tmp, video_path)

        stationary = [100.0, 100.0, 130.0, 180.0]
        tracklet_rows = [
            {"tracklet_id": 7, "frame_no": f, "det_id": f, "xyxy": stationary} for f in range(100)
        ]
        # P3 mapped tracklet 7 → person_id 4.
        persons = [
            {
                "person_id": 4,
                "tracklet_ids": [7],
                "embedding_centroids": {},
                "embedding_counts": {},
                "first_seen": 0.0,
                "last_seen": 9.9,
                "confirmation_state": "confirmed",
                "assoc_audit": [],
            }
        ]
        _seed_p1_p2_p3(store, frames_processed=100, tracklet_rows=tracklet_rows, persons=persons)

        config = OfflineConfig()
        orch = OfflineOrchestrator(meta, store, config)
        orch.run_pass_p5()

        events = list(store.iter_events(OfflineOrchestrator.P5_PASS_NAME))
        stilla = [e for e in events if e["category"] == "STILLA"]
        assert stilla
        assert all(e["person_id"] == 4 for e in stilla)


def test_run_pass_p5_person_id_null_when_p3_skipped():
    """Without P3, person-keyed events get person_id=null (per the report's
    events/ schema, person_id is null where not applicable)."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "v.mp4"
        _make_video(video_path, n_frames=100, fps=10.0)
        meta, store = _make_meta_and_store(tmp, video_path)

        stationary = [100.0, 100.0, 130.0, 180.0]
        tracklet_rows = [
            {"tracklet_id": 1, "frame_no": f, "det_id": f, "xyxy": stationary} for f in range(100)
        ]
        _seed_p1_p2_p3(store, frames_processed=100, tracklet_rows=tracklet_rows, persons=[])

        config = OfflineConfig()
        orch = OfflineOrchestrator(meta, store, config)
        orch.run_pass_p5()

        events = list(store.iter_events(OfflineOrchestrator.P5_PASS_NAME))
        assert all(e["person_id"] is None for e in events)
        # Manifest records that P3 was not used.
        p5_stats = store.manifest["passes"][OfflineOrchestrator.P5_PASS_NAME]["stats"]
        assert p5_stats["p3_used"] is False


def test_run_pass_p5_refuses_when_p2_incomplete():
    """P5 gates on P2 success — refusing to run over an incomplete artifact
    prevents an events/ table built from a truncated tracklet set that the
    manifest would present as final."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "v.mp4"
        _make_video(video_path, n_frames=20, fps=10.0)
        meta, store = _make_meta_and_store(tmp, video_path)

        # Record P2 as failed (no tracklets written).
        store.record_pass_start(OfflineOrchestrator.P2_PASS_NAME, {})
        store.record_pass_error(OfflineOrchestrator.P2_PASS_NAME, "simulated crash")

        config = OfflineConfig()
        orch = OfflineOrchestrator(meta, store, config)
        n = orch.run_pass_p5()
        assert n == 0
        p5 = store.manifest["passes"][OfflineOrchestrator.P5_PASS_NAME]
        assert p5["status"] == "error"
        assert "P2 not complete" in p5["error"]


def test_run_pass_p5_truncates_previous_output():
    """P5 always re-runs in full like P2/P3 — a second invocation must
    overwrite (not append to) the previous events/ output."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "v.mp4"
        _make_video(video_path, n_frames=100, fps=10.0)
        meta, store = _make_meta_and_store(tmp, video_path)

        stationary = [100.0, 100.0, 130.0, 180.0]
        tracklet_rows = [
            {"tracklet_id": 1, "frame_no": f, "det_id": f, "xyxy": stationary} for f in range(100)
        ]
        _seed_p1_p2_p3(store, frames_processed=100, tracklet_rows=tracklet_rows, persons=[])

        config = OfflineConfig()
        orch = OfflineOrchestrator(meta, store, config)
        first_n = orch.run_pass_p5()
        # Inject a stray row to simulate stale output from a prior invocation.
        with open(store.run_dir / "events" / f"{OfflineOrchestrator.P5_PASS_NAME}.jsonl", "a") as f:
            f.write('{"event_id":"stale","category":"STILLA","person_id":null}\n')
        second_n = orch.run_pass_p5()
        assert second_n == first_n  # same input → same output
        events = list(store.iter_events(OfflineOrchestrator.P5_PASS_NAME))
        assert all(e["event_id"] != "stale" for e in events)
        assert len(events) == second_n
