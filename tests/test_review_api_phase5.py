"""API tests for Phase 5: feature toggles + the five new surfaces.

Every feature is exercised in BOTH toggle states (the exit criterion asks
for exactly that): default-on behavior, explicit off (404 naming the env
var), and independence (turning one off leaves the others working). Plus
the endpoint semantics themselves: identity corrections (write validation,
projection into /persons, /tracklets, /events, undo), ground truth
(import/score), run comparison, heatmap, and the trajectory endpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from analysis.orchestrator import OfflineOrchestrator
from analysis.store import ArtifactStore
from review.config import ReviewSettings
from review.main import app

P1 = OfflineOrchestrator.P1_PASS_NAME
P2 = OfflineOrchestrator.P2_PASS_NAME
P3 = OfflineOrchestrator.P3_PASS_NAME
P5 = OfflineOrchestrator.P5_PASS_NAME

ALL_FLAGS = (
    "FEATURE_DOSSIER",
    "FEATURE_GROUND_TRUTH",
    "FEATURE_RUN_COMPARE",
    "FEATURE_CLIP_EXPORT",
    "FEATURE_HEATMAP",
)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch) -> ReviewSettings:
    output_dir = tmp_path / "analysis-output"
    output_dir.mkdir()
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "film.mp4").write_bytes(b"FAKE-MP4-BYTES")
    monkeypatch.setenv("ANALYSIS_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("VIDEO_DIR", str(video_dir))
    for flag in ALL_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    return ReviewSettings.from_env()


def seed_run(
    settings: ReviewSettings, video_hash: str = "vh-aaa", tiles: int = 1, scene: bool = False
) -> str:
    """A run with two persons (one two-tracklet), P1 dims, and P5 events."""
    store = ArtifactStore(str(settings.output_dir), video_hash, f"ch-{tiles}")
    store.create()
    store.set_video_filename("film.mp4")
    store.record_pass_start(
        P1, {"fps": 10.0, "config": {"tiles": tiles, "model": "yolo11n.pt"}, "width": 1000, "height": 500}
    )
    for fn in range(10):
        store.add_frame(P1, fn, {"pts_ms": fn * 100.0})
    store.record_pass_complete(P1, {"frames_processed": 10, "total_detections": 20})
    store.record_pass_start(P2, {})
    store.start_fresh_pass_output("tracklets", P2)
    if scene:
        store.start_fresh_pass_output("frames", P2)
        for fn in range(10):
            tx = float(fn * 5)
            store.add_frame(
                P2,
                fn,
                {
                    "pts_ms": fn * 100.0,
                    "scene_segment": 0,
                    "scene_to_frame": [[1.0, 0.0, tx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "frame_to_scene": [[1.0, 0.0, -tx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "scene_confidence": 0.9,
                    "scene_linked": True,
                },
            )

    def tracklet_payload(x0: float, y0: float, x1: float, y1: float, fn: int) -> dict:
        payload = {"cls": "person", "conf": 0.9, "xyxy": [x0, y0, x1, y1]}
        if scene:
            payload.update(
                {
                    "scene_pos": [(x0 + x1) / 2.0 - fn * 5, y1],
                    "scene_box_h": y1 - y0,
                    "scene_segment": 0,
                }
            )
        return payload

    # Tracklet 1: frames 0-4, tracklet 3: frames 6-9 (same person per P3),
    # tracklet 2: frames 0-9 (the other person).
    for fn in range(5):
        store.add_tracklet_frame(P2, 1, fn, fn * 3, tracklet_payload(100.0, 100.0, 130.0, 200.0, fn))
    for fn in range(6, 10):
        store.add_tracklet_frame(P2, 3, fn, fn * 3 + 1, tracklet_payload(120.0, 100.0, 150.0, 200.0, fn))
    for fn in range(10):
        store.add_tracklet_frame(P2, 2, fn, fn * 3 + 2, tracklet_payload(800.0, 300.0, 830.0, 400.0, fn))
    store.record_pass_complete(P2, {"total_tracklet_rows": 19})
    store.record_pass_start(P3, {})
    store.start_fresh_pass_output("persons", P3)
    store.add_person(
        P3,
        1,
        {
            "tracklet_ids": [1, 3],
            "embedding_centroids": {"hsv": [1.0, 0.0]},
            "embedding_counts": {"hsv": 9},
            "first_seen": 0.0,
            "last_seen": 0.9,
            "confirmation_state": "confirmed",
            "assoc_audit": [{"tracklet_a": 1, "tracklet_b": 3, "rule": "merged"}],
        },
    )
    store.add_person(
        P3,
        2,
        {
            "tracklet_ids": [2],
            "embedding_centroids": {"hsv": [0.0, 1.0]},
            "embedding_counts": {"hsv": 10},
            "first_seen": 0.0,
            "last_seen": 0.9,
            "confirmation_state": "confirmed",
            "assoc_audit": [],
        },
    )
    store.record_pass_complete(P3, {"persons_out": 2, "confirmed_persons": 2})
    store.record_pass_start(P5, {"fps": 10.0, "config": {"tiles": tiles}})
    store.start_fresh_pass_output("events", P5)
    store.add_event(
        P5,
        "stilla-000000",
        {
            "category": "STILLA",
            "person_id": 1,
            "t_start": 0.6,
            "t_end": 0.9,
            "confidence": 0.7,
            "evidence": {"tracklet_id": 3},
            "review": {"state": "unreviewed"},
        },
    )
    store.record_pass_complete(P5, {"events_out": 1, "by_category": {"STILLA": 1}, "p3_used": True})
    store.close()
    return store.run_id


def seed_counts_run(settings: ReviewSettings) -> str:
    """A run whose persons cover every confirmation state the unique-count
    headline has to distinguish: two confirmed (mergeable into one manual)
    plus one more confirmed and one transient."""
    store = ArtifactStore(str(settings.output_dir), "vh-counts", "ch-counts")
    store.create()
    store.set_video_filename("film.mp4")
    store.record_pass_start(P1, {"fps": 10.0, "config": {}, "width": 1000, "height": 500})
    store.record_pass_complete(P1, {"frames_processed": 4})
    store.record_pass_start(P2, {})
    store.start_fresh_pass_output("tracklets", P2)
    for tid in (1, 2, 3, 4):
        for fn in range(4):
            store.add_tracklet_frame(
                P2, tid, fn, tid * 10 + fn, {"cls": "person", "conf": 0.9, "xyxy": [10.0, 10.0, 40.0, 110.0]}
            )
    store.record_pass_complete(P2, {"total_tracklet_rows": 16})
    store.record_pass_start(P3, {})
    store.start_fresh_pass_output("persons", P3)
    for pid, tids, conf_state in (
        (1, [1], "confirmed"),
        (2, [2], "confirmed"),
        (3, [3], "confirmed"),
        (4, [4], "transient"),
    ):
        store.add_person(
            P3,
            pid,
            {
                "tracklet_ids": tids,
                "embedding_centroids": {},
                "embedding_counts": {},
                "first_seen": 0.0,
                "last_seen": 0.3,
                "confirmation_state": conf_state,
                "assoc_audit": [],
            },
        )
    store.record_pass_complete(P3, {"persons_out": 4, "confirmed_persons": 3, "uncertain_merges": 2})
    store.close()
    return store.run_id


@pytest.fixture
def run_id(settings: ReviewSettings) -> str:
    return seed_run(settings)


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ---- guide page ----


async def test_guide_page_served_in_swedish(client):
    r = await client.get("/guide")
    assert r.status_code == 200
    assert "Användarhandledning" in r.text
    # Covers the core workflows by their Swedish section names.
    for heading in ("Analysera en film", "Faromarkören", "Facit", "Värmekartan"):
        assert heading in r.text


async def test_index_links_to_guide(client):
    r = await client.get("/")
    assert 'href="/guide"' in r.text


# ---- toggle state surface ----


async def test_features_default_all_on(settings, client):
    r = await client.get("/api/features")
    assert r.status_code == 200
    assert r.json() == {
        "dossier": True,
        "ground_truth": True,
        "run_compare": True,
        "clip_export": True,
        "heatmap": True,
    }


async def test_features_reflect_env(settings, client, monkeypatch):
    monkeypatch.setenv("FEATURE_CLIP_EXPORT", "0")
    monkeypatch.setenv("FEATURE_HEATMAP", "off")
    r = await client.get("/api/features")
    body = r.json()
    assert body["clip_export"] is False
    assert body["heatmap"] is False
    # Independence: the others stay on.
    assert body["dossier"] is True and body["ground_truth"] is True and body["run_compare"] is True


async def test_garbled_env_value_falls_back_to_default(settings, client, monkeypatch):
    monkeypatch.setenv("FEATURE_DOSSIER", "banan")
    r = await client.get("/api/features")
    assert r.json()["dossier"] is True


# ---- gating: each feature off → 404 naming the env var; others unaffected ----


async def test_dossier_off_gates_writes_but_not_reads(settings, run_id, client, monkeypatch):
    monkeypatch.setenv("FEATURE_DOSSIER", "0")
    r = await client.post(
        f"/api/runs/{run_id}/identity-corrections", data={"op": "merge", "person_ids": "1,2"}
    )
    assert r.status_code == 404
    assert "FEATURE_DOSSIER" in r.json()["detail"]
    r = await client.get(f"/api/runs/{run_id}/persons/1/trajectory")
    assert r.status_code == 404
    # The corrections LIST stays readable (recorded human work is never hidden).
    r = await client.get(f"/api/runs/{run_id}/identity-corrections")
    assert r.status_code == 200
    # Other features still work.
    assert (await client.get(f"/api/runs/{run_id}/heatmap")).status_code == 200
    assert (await client.get(f"/api/runs/{run_id}/ground-truth")).status_code == 200


async def test_ground_truth_off(settings, run_id, client, monkeypatch):
    monkeypatch.setenv("FEATURE_GROUND_TRUTH", "0")
    for method, url, kwargs in (
        ("post", f"/api/runs/{run_id}/ground-truth/import", {"data": {"text": "0:10, x"}}),
        ("get", f"/api/runs/{run_id}/ground-truth", {}),
        ("get", f"/api/runs/{run_id}/ground-truth/score", {}),
        ("delete", f"/api/runs/{run_id}/ground-truth/abc123", {}),
    ):
        r = await getattr(client, method)(url, **kwargs)
        assert r.status_code == 404, url
        assert "FEATURE_GROUND_TRUTH" in r.json()["detail"]
    # Independence.
    assert (await client.get(f"/api/runs/{run_id}/heatmap")).status_code == 200


async def test_run_compare_off(settings, run_id, client, monkeypatch):
    monkeypatch.setenv("FEATURE_RUN_COMPARE", "0")
    r = await client.get(f"/api/runs/{run_id}/compare/{run_id}")
    assert r.status_code == 404
    assert "FEATURE_RUN_COMPARE" in r.json()["detail"]


async def test_heatmap_off(settings, run_id, client, monkeypatch):
    monkeypatch.setenv("FEATURE_HEATMAP", "0")
    r = await client.get(f"/api/runs/{run_id}/heatmap")
    assert r.status_code == 404
    assert "FEATURE_HEATMAP" in r.json()["detail"]
    # Independence.
    assert (await client.get(f"/api/runs/{run_id}/persons/1/trajectory")).status_code == 200


# ---- identity corrections ----


async def test_merge_projects_into_persons_tracklets_events(settings, run_id, client):
    r = await client.post(
        f"/api/runs/{run_id}/identity-corrections", data={"op": "merge", "person_ids": "1,2"}
    )
    assert r.status_code == 201
    body = r.json()
    assert body["correction"]["op"] == "merge"
    # Persons 1+2 overlap in time (both span the whole clip) → warning.
    assert body["warning"] is not None

    r = await client.get(f"/api/runs/{run_id}/persons")
    persons = r.json()
    assert persons["count"] == 1
    assert persons["corrections_applied"] == 1
    rec = persons["persons"][0]
    assert rec["person_id"] == 1
    assert rec["tracklet_ids"] == [1, 2, 3]
    assert rec["confirmation_state"] == "manual"
    assert rec["corrected"] is True

    # Overlay join follows the correction.
    r = await client.get(f"/api/runs/{run_id}/tracklets", params={"frame": 0})
    boxes = {b["tracklet_id"]: b["person_id"] for b in r.json()["boxes"]}
    assert boxes[2] == 1

    # Served events follow their tracklet to the corrected person.
    r = await client.get(f"/api/runs/{run_id}/events")
    assert r.json()["events"][0]["person_id"] == 1


async def test_split_creates_new_person_and_remaps_event(settings, run_id, client):
    r = await client.post(
        f"/api/runs/{run_id}/identity-corrections",
        data={"op": "split", "person_id": "1", "tracklet_ids": "3", "reason": "olika kläder"},
    )
    assert r.status_code == 201
    r = await client.get(f"/api/runs/{run_id}/persons")
    persons = {p["person_id"]: p for p in r.json()["persons"]}
    assert set(persons) == {1, 2, 3}
    assert persons[1]["tracklet_ids"] == [1]
    assert persons[3]["tracklet_ids"] == [3]
    # The STILLA event rides tracklet 3 → now person 3.
    r = await client.get(f"/api/runs/{run_id}/events")
    assert r.json()["events"][0]["person_id"] == 3


async def test_persons_unique_count_excludes_transients(settings, client):
    """count = every projected row; unique_count = confirmed + manual only.

    The `manual` person is produced by actually REPLAYING a merge op (not by
    hand-stamping confirmation_state on a fixture row), so the assertion can
    fail if the keying or the projection breaks.
    """
    rid = seed_counts_run(settings)
    r = await client.get(f"/api/runs/{rid}/persons")
    body = r.json()
    assert body["count"] == 4
    assert body["unique_count"] == 3  # persons 1-3 confirmed, person 4 transient
    engine_uncertainty = {
        "run_id": rid,
        "pass": P3,
        "uncertain_merges": 2,
    }
    assert body["engine_uncertainty"] == engine_uncertainty

    r = await client.post(f"/api/runs/{rid}/identity-corrections", data={"op": "merge", "person_ids": "1,2"})
    assert r.status_code == 201

    body = (await client.get(f"/api/runs/{rid}/persons")).json()
    states = {p["person_id"]: p["confirmation_state"] for p in body["persons"]}
    assert states == {1: "manual", 3: "confirmed", 4: "transient"}
    assert body["count"] == 3
    assert body["unique_count"] == 2
    # A human merge changes the live projection, not P3's historical evidence.
    assert body["engine_uncertainty"] == engine_uncertainty


async def test_persons_engine_uncertainty_defaults_for_legacy_sidecar(settings, run_id, client):
    body = (await client.get(f"/api/runs/{run_id}/persons")).json()
    assert body["engine_uncertainty"] == {
        "run_id": run_id,
        "pass": P3,
        "uncertain_merges": 0,
    }


async def test_correction_row_records_tracklet_key_and_provenance(settings, run_id, client):
    r = await client.post(
        f"/api/runs/{run_id}/identity-corrections", data={"op": "merge", "person_ids": "1,2"}
    )
    row = r.json()["correction"]
    # The op's real key: each member's full tracklet set at correction time.
    assert row["member_tracklet_ids"] == [[1, 3], [2]]
    # person_ids stay as provenance, alongside the run's hashes.
    assert row["person_ids"] == [1, 2]
    assert row["video_hash"] == "vh-aaa" and row["config_hash"] == "ch-1"

    r = await client.post(
        f"/api/runs/{run_id}/identity-corrections",
        data={"op": "split", "person_id": "1", "tracklet_ids": "3"},
    )
    row = r.json()["correction"]
    assert row["source_tracklet_ids"] == [1, 2, 3]  # the merged person's full set
    assert row["tracklet_ids"] == [3]


async def test_correction_keyed_by_tracklets_not_person_id_after_renumber(settings, run_id, client):
    """A recorded merge must never land on a different pair of people just
    because a re-analysis reused the same positional person ids."""
    r = await client.post(
        f"/api/runs/{run_id}/identity-corrections", data={"op": "merge", "person_ids": "1,2"}
    )
    assert r.status_code == 201

    # Re-analysis: same contiguous ids 1..2, different tracklet grouping.
    store = ArtifactStore.open_readonly(settings.output_dir / run_id)
    persons_path = store.run_dir / "persons" / f"{P3}.jsonl"
    persons_path.write_text(
        "\n".join(
            [
                '{"person_id":1,"tracklet_ids":[1],"embedding_centroids":{},"embedding_counts":{},'
                '"first_seen":0.0,"last_seen":0.4,"confirmation_state":"confirmed","assoc_audit":[]}',
                '{"person_id":2,"tracklet_ids":[2,3],"embedding_centroids":{},"embedding_counts":{},'
                '"first_seen":0.0,"last_seen":0.9,"confirmation_state":"confirmed","assoc_audit":[]}',
            ]
        )
        + "\n"
    )

    body = (await client.get(f"/api/runs/{run_id}/persons")).json()
    assert body["corrections_applied"] == 0
    assert body["count"] == 2
    assert [p["tracklet_ids"] for p in body["persons"]] == [[1], [2, 3]]
    assert all(p["confirmation_state"] == "confirmed" for p in body["persons"])
    assert "motsvarar inte längre exakt en person" in body["corrections_skipped"][0]["reason"]


async def test_invalid_correction_rejected_not_persisted(settings, run_id, client):
    r = await client.post(
        f"/api/runs/{run_id}/identity-corrections", data={"op": "merge", "person_ids": "1,99"}
    )
    assert r.status_code == 422
    assert "99" in r.json()["detail"]
    r = await client.get(f"/api/runs/{run_id}/identity-corrections")
    assert r.json()["corrections"] == []


async def test_undo_correction(settings, run_id, client):
    r = await client.post(
        f"/api/runs/{run_id}/identity-corrections", data={"op": "merge", "person_ids": "1,2"}
    )
    aid = r.json()["correction"]["annotation_id"]
    assert (await client.get(f"/api/runs/{run_id}/persons")).json()["count"] == 1
    r = await client.delete(f"/api/runs/{run_id}/identity-corrections/{aid}")
    assert r.status_code == 200
    assert (await client.get(f"/api/runs/{run_id}/persons")).json()["count"] == 2


async def test_corrections_still_applied_when_dossier_toggled_off(settings, run_id, client, monkeypatch):
    # Record with the feature on...
    await client.post(f"/api/runs/{run_id}/identity-corrections", data={"op": "merge", "person_ids": "1,2"})
    # ...then turn it off: the recorded correction keeps applying.
    monkeypatch.setenv("FEATURE_DOSSIER", "0")
    r = await client.get(f"/api/runs/{run_id}/persons")
    assert r.json()["count"] == 1


async def test_trajectory_endpoint(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/persons/1/trajectory")
    assert r.status_code == 200
    body = r.json()
    # Person 1 owns tracklets 1 (5 frames) + 3 (4 frames).
    assert body["n_total"] == 9
    assert body["stride"] == 1
    assert body["coordinate_space"] == "frame"
    assert body["points"][0] == {"frame_no": 0, "t": 0.0, "x": 115.0, "y": 200.0, "tracklet_id": 1}
    r = await client.get(f"/api/runs/{run_id}/persons/42/trajectory")
    assert r.status_code == 404


async def test_trajectory_endpoint_uses_scene_coordinates(settings, client):
    scene_run = seed_run(settings, video_hash="vh-scene", tiles=3, scene=True)
    r = await client.get(f"/api/runs/{scene_run}/persons/1/trajectory")

    assert r.status_code == 200
    body = r.json()
    assert body["coordinate_space"] == "scene"
    assert body["points"][0]["scene_segment"] == 0
    # Raw foot x is 115 in every frame, but the camera translates +5/frame.
    assert body["points"][0]["x"] == 115.0
    assert body["points"][1]["x"] == 110.0


# ---- ground truth ----


async def test_ground_truth_import_and_score(settings, run_id, client):
    r = await client.post(
        f"/api/runs/{run_id}/ground-truth/import",
        data={"text": "0:01, figurant ligger vid bilen\nstrunt utan tid\n8:00, figurant två"},
    )
    assert r.status_code == 201
    body = r.json()
    assert len(body["imported"]) == 2
    assert len(body["warnings"]) == 1  # the untimed line

    # Operator note near the first reference entry.
    await client.post(f"/api/runs/{run_id}/operator-notes/import", data={"text": "0:05, ser en person"})

    r = await client.get(f"/api/runs/{run_id}/ground-truth/score")
    assert r.status_code == 200
    score = r.json()
    assert score["counts"]["gt_total"] == 2
    # AI's STILLA at t=0.6 matches the 0:01 entry; operator's 0:05 note too.
    assert score["counts"]["ai_found"] == 1
    assert score["counts"]["operator_found"] == 1
    assert score["counts"]["ai_missed"] == 1
    first = score["entries"][0]
    assert first["ai"]["event"]["event_id"] == "stilla-000000"
    assert first["operator"]["note"]["text"] == "ser en person"
    # The 8:00 entry is missed by both.
    assert score["entries"][1]["ai"] is None and score["entries"][1]["operator"] is None


async def test_ground_truth_delete(settings, run_id, client):
    r = await client.post(f"/api/runs/{run_id}/ground-truth/import", data={"text": "0:01, x"})
    aid = r.json()["imported"][0]["annotation_id"]
    assert (await client.delete(f"/api/runs/{run_id}/ground-truth/{aid}")).status_code == 200
    assert (await client.get(f"/api/runs/{run_id}/ground-truth")).json()["count"] == 0


# ---- run comparison ----


async def test_run_compare_endpoint(settings, run_id, client):
    other = seed_run(settings, video_hash="vh-aaa", tiles=2)
    r = await client.get(f"/api/runs/{run_id}/compare/{other}")
    assert r.status_code == 200
    body = r.json()
    assert body["config_diff"] == {"tiles": {"a": 1, "b": 2}}
    assert body["counts"]["both"] == 1  # identical STILLA event in both


async def test_run_compare_refuses_other_video(settings, run_id, client):
    other = seed_run(settings, video_hash="vh-bbb")
    r = await client.get(f"/api/runs/{run_id}/compare/{other}")
    assert r.status_code == 409
    assert "video" in r.json()["detail"]


# ---- heatmap ----


async def test_heatmap_endpoint(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/heatmap")
    assert r.status_code == 200
    body = r.json()
    # Dims come from P1's recorded pass meta.
    assert body["frame_w"] == 1000 and body["frame_h"] == 500
    assert body["grid_w"] == 48
    # 19 tracklet rows at 10 fps.
    assert body["total_s"] == pytest.approx(1.9)


async def test_heatmap_person_filter_respects_corrections(settings, run_id, client):
    r = await client.get(f"/api/runs/{run_id}/heatmap", params={"person_id": 1})
    assert r.json()["total_s"] == pytest.approx(0.9)  # tracklets 1+3 only
    # After merging person 2 into 1, the filter covers all rows.
    await client.post(f"/api/runs/{run_id}/identity-corrections", data={"op": "merge", "person_ids": "1,2"})
    r = await client.get(f"/api/runs/{run_id}/heatmap", params={"person_id": 1})
    assert r.json()["total_s"] == pytest.approx(1.9)
