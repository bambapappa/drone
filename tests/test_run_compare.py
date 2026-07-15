"""Tests for the multi-config run diff (Phase 5, report §5.7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.orchestrator import OfflineOrchestrator
from analysis.store import ArtifactStore
from review.run_compare import RunCompareError, compare_runs, main

P1 = OfflineOrchestrator.P1_PASS_NAME
P5 = OfflineOrchestrator.P5_PASS_NAME


def seed_run(
    output_dir: Path,
    video_hash: str,
    config: dict,
    events: list[dict],
    stats: dict | None = None,
) -> ArtifactStore:
    store = ArtifactStore(str(output_dir), video_hash, "ch-" + video_hash)
    store.create()
    store.record_pass_start(P1, {"config": config, "fps": 25.0})
    store.record_pass_complete(P1, {"total_detections": stats.get("dets", 0) if stats else 0})
    store.record_pass_start(P5, {"config": config, "fps": 25.0})
    store.start_fresh_pass_output("events", P5)
    for ev in events:
        store.add_event(P5, ev["event_id"], ev)
    by_cat: dict[str, int] = {}
    for ev in events:
        by_cat[ev["category"]] = by_cat.get(ev["category"], 0) + 1
    store.record_pass_complete(P5, {"events_out": len(events), "by_category": by_cat})
    store.close()
    return store


def ev(eid, category, t_start, t_end=None):
    return {
        "event_id": eid,
        "category": category,
        "person_id": None,
        "t_start": t_start,
        "t_end": t_end if t_end is not None else t_start + 4.0,
        "confidence": 0.5,
        "evidence": {},
        "review": {"state": "unreviewed"},
    }


def reopen(store: ArtifactStore) -> ArtifactStore:
    return ArtifactStore.open_readonly(store.run_dir)


def test_refuses_different_videos(tmp_path):
    a = seed_run(tmp_path, "hash-a", {}, [])
    b = seed_run(tmp_path, "hash-b", {}, [])
    with pytest.raises(RunCompareError, match="video_hash"):
        compare_runs(reopen(a), reopen(b))


def test_refuses_missing_p5(tmp_path):
    a = seed_run(tmp_path, "hash-x", {}, [])
    b = ArtifactStore(str(tmp_path), "hash-x", "ch")
    b.create()
    b.close()
    with pytest.raises(RunCompareError, match="P5"):
        compare_runs(reopen(a), reopen(b))


def test_config_diff_and_buckets(tmp_path):
    cfg_a = {"model": "yolo11n.pt", "tiles": 1, "imgsz": 640}
    cfg_b = {"model": "visdrone.pt", "tiles": 2, "imgsz": 640}
    a = seed_run(tmp_path, "h", cfg_a, [ev("stilla-000000", "STILLA", 10.0)], {"dets": 5})
    b = seed_run(
        tmp_path,
        "h",
        cfg_b,
        [ev("stilla-000000", "STILLA", 12.0), ev("hazard-000000", "HAZARD", 40.0)],
        {"dets": 50},
    )
    result = compare_runs(reopen(a), reopen(b), tolerance_s=10.0)
    # Config diff names exactly the changed fields, with both values.
    assert result["config_diff"] == {
        "model": {"a": "yolo11n.pt", "b": "visdrone.pt"},
        "tiles": {"a": 1, "b": 2},
    }
    # The STILLA pair matches (Δ=2s ≤ 10s); B's HAZARD is B-only.
    assert result["counts"] == {"both": 1, "only_a": 0, "only_b": 1}
    assert result["both"][0]["delta_s"] == 2.0
    assert result["only_b"][0]["event_id"] == "hazard-000000"
    assert result["stats"]["a"]["detections"] == 5
    assert result["stats"]["b"]["detections"] == 50


def test_no_cross_category_matching(tmp_path):
    # Same onset, different categories → never a correspondence.
    a = seed_run(tmp_path, "h2", {}, [ev("stilla-000000", "STILLA", 10.0)])
    b = seed_run(tmp_path, "h2", {}, [ev("hazard-000000", "HAZARD", 10.0)])
    result = compare_runs(reopen(a), reopen(b), tolerance_s=10.0)
    assert result["counts"] == {"both": 0, "only_a": 1, "only_b": 1}


def test_deterministic(tmp_path):
    events = [ev(f"stilla-{i:06d}", "STILLA", 10.0 * i) for i in range(4)]
    a = seed_run(tmp_path, "h3", {"tiles": 1}, events)
    b = seed_run(tmp_path, "h3", {"tiles": 2}, events[:2])
    r1 = compare_runs(reopen(a), reopen(b))
    r2 = compare_runs(reopen(a), reopen(b))
    assert r1 == r2


def test_cli_summary_and_toggle(tmp_path, capsys, monkeypatch):
    a = seed_run(tmp_path, "h4", {"tiles": 1}, [ev("stilla-000000", "STILLA", 1.0)])
    b = seed_run(tmp_path, "h4", {"tiles": 2}, [ev("stilla-000000", "STILLA", 2.0)])
    assert main([str(a.run_dir), str(b.run_dir)]) == 0
    out = capsys.readouterr().out
    assert "tiles" in out and "gemensamma" in out
    # The CLI honors the same toggle as the REST surface.
    monkeypatch.setenv("FEATURE_RUN_COMPARE", "0")
    assert main([str(a.run_dir), str(b.run_dir)]) == 2


def test_cli_error_on_video_mismatch(tmp_path, capsys):
    a = seed_run(tmp_path, "h5", {}, [])
    b = seed_run(tmp_path, "h6", {}, [])
    assert main([str(a.run_dir), str(b.run_dir)]) == 2
    assert "video" in capsys.readouterr().err
