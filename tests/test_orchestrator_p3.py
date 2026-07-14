"""Acceptance tests for P3 (identity pass) at the orchestrator level.

Mirrors Phase 0's P2 determinism test pattern: P3 is never checkpointed and
always re-runs in full, so its correctness guarantee is that two runs over the
same P1+P2 output produce byte-identical persons/assoc_audit. Driven through
a FakeDetector + a deterministic FakeEmbedder (no torch/weights), the same way
Phase 0's P1 tests use FakeDetector — the semantic clustering logic is covered
in test_identity.py; here we assert the end-to-end P1→P2→P3 wiring and its
determinism, plus the P2-success gate.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np

from analysis.detector import Detection
from analysis.ingest import VideoMeta, build_pts_index
from analysis.orchestrator import OfflineConfig, OfflineOrchestrator
from analysis.store import ArtifactStore


def _make_test_video(path: str, n_frames: int = 12, fps: float = 10.0) -> str:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (64, 64))
    for i in range(n_frames):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:] = (60, 90, 120)
        cv2.rectangle(frame, (5 + i, 5), (40 + i, 50), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def _make_meta(video_path: str) -> VideoMeta:
    index, fps, w, h, total = build_pts_index(video_path)
    with open(video_path, "rb") as f:
        vhash = hashlib.sha256(f.read(100_000)).hexdigest()
    return VideoMeta(
        path=str(Path(video_path).resolve()),
        fps=fps,
        width=w,
        height=h,
        total_frames=total,
        video_hash=vhash,
        pts_index=index,
    )


class FakeDetector:
    """Stateless stand-in for the real YOLO Detector (P1 has no tracker to
    fake). Produces a deterministic human detection per frame so P1+P2 yield
    a tracklet for P3 to cluster over."""

    def __init__(self, **kwargs):
        pass

    def detect(self, frame):
        h, w = frame.shape[:2]
        # One moving "person" box; position depends only on frame pixels so
        # it is deterministic across runs (no RNG consumed in eval).
        return [
            Detection(
                track_id=None,
                cls_name="person",
                cls_id=0,
                conf=0.9,
                xyxy=(10.0, 10.0, 30.0, 50.0),
                is_human=True,
                is_threat=False,
            )
        ]


def _fake_make_embedder(_config):
    """Deterministic embedder: encodes the crop's mean BGR into a fixed-dim
    normalized vector tagged 'hsv'. No torch, no weights — lets P3 run over
    real P1+P2 output in CI. The embedding is a pure function of the frame, so
    two runs produce identical embeddings -> byte-identical P3 output."""
    from analysis.embedding import EmbeddingResult

    class _Fake:
        def embed(self, frame_bgr, box_xyxy):
            x0, y0, x1, y1 = (int(v) for v in box_xyxy)
            crop = frame_bgr[max(0, y0) : y1, max(0, x0) : x1]
            if crop.size == 0:
                return None
            mean = crop.reshape(-1, 3).mean(axis=0)
            v = np.asarray(mean, dtype=np.float64)
            n = np.linalg.norm(v)
            return EmbeddingResult(vector=v / n if n > 0 else v, method="hsv")

    return _Fake()


def _persons_bytes(store: ArtifactStore) -> bytes:
    p = store.run_dir / "persons" / f"{OfflineOrchestrator.P3_PASS_NAME}.jsonl"
    return p.read_bytes() if p.exists() else b""


def _run_full(orch: OfflineOrchestrator, monkeypatch) -> None:
    monkeypatch.setattr("analysis.detector.Detector", FakeDetector)
    monkeypatch.setattr("analysis.embedding.make_embedder", _fake_make_embedder)
    orch.run_pass_p1()
    orch.run_pass_p2()
    orch.run_pass_p3()


class TestP3Determinism:
    def test_two_full_runs_produce_byte_identical_persons(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4")
            meta = _make_meta(vpath)
            config = OfflineConfig(seed=42)

            store_a = ArtifactStore(f"{tmp}/out_a", meta.video_hash, "cfg")
            store_a.create()
            _run_full(OfflineOrchestrator(meta, store_a, config), monkeypatch)

            store_b = ArtifactStore(f"{tmp}/out_b", meta.video_hash, "cfg")
            store_b.create()
            _run_full(OfflineOrchestrator(meta, store_b, config), monkeypatch)

            a = _persons_bytes(store_a)
            b = _persons_bytes(store_b)
            assert a == b
            assert a  # sanity: P3 actually produced a persons table

    def test_p3_rerun_overwrites_not_appends(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4")
            meta = _make_meta(vpath)
            config = OfflineConfig(seed=42)

            store = ArtifactStore(f"{tmp}/out", meta.video_hash, "cfg")
            store.create()
            orch = OfflineOrchestrator(meta, store, config)
            _run_full(orch, monkeypatch)
            first = _persons_bytes(store)

            # Re-run P3 alone (P1/P2 already complete on the store).
            orch.run_pass_p3()
            second = _persons_bytes(store)

            assert second == first  # no duplicate rows, identical content


class TestP3GatingAndShape:
    def test_p3_refuses_when_p2_incomplete(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=6)
            meta = _make_meta(vpath)
            config = OfflineConfig(seed=1)

            store = ArtifactStore(f"{tmp}/out", meta.video_hash, "cfg")
            store.create()
            orch = OfflineOrchestrator(meta, store, config)
            monkeypatch.setattr("analysis.detector.Detector", FakeDetector)
            monkeypatch.setattr("analysis.embedding.make_embedder", _fake_make_embedder)
            orch.run_pass_p1()
            # NOTE: deliberately do NOT run P2.
            result = orch.run_pass_p3()
            assert result is None
            assert _persons_bytes(store) == b""  # nothing written
            p3_info = store._manifest["passes"][OfflineOrchestrator.P3_PASS_NAME]
            assert p3_info["status"] == "error"

    def test_p3_writes_persons_table_with_audit(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4")
            meta = _make_meta(vpath)
            config = OfflineConfig(seed=3)

            store = ArtifactStore(f"{tmp}/out", meta.video_hash, "cfg")
            store.create()
            _run_full(OfflineOrchestrator(meta, store, config), monkeypatch)

            persons = list(store.iter_persons(OfflineOrchestrator.P3_PASS_NAME))
            assert len(persons) >= 1
            for p in persons:
                assert "person_id" in p
                assert "tracklet_ids" in p
                assert "assoc_audit" in p  # load-bearing, always present
                assert "embedding_centroids" in p
                assert "first_seen" in p and "last_seen" in p
                assert p["confirmation_state"] in ("confirmed", "transient")
            # manifest stats recorded
            stats = store._manifest["passes"][OfflineOrchestrator.P3_PASS_NAME]["stats"]
            assert "confirmed_persons" in stats
            assert "uncertain_merges" in stats
