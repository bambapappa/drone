"""Acceptance tests for explicit --resume, seeded reproducibility (P1), and
P2's full-re-run determinism.

P1 is stateless (no tracker), so these tests use a fake, dependency-free
Detector (no real YOLO/torch) to exercise the orchestrator/store checkpoint-
resume and per-frame seeding machinery without ML weights. P2 owns the real
tracking logic, so its determinism test drives the real BoT-SORT tracker
(analysis.tracker.Tracker) instead of a fake — a fake tracker would prove
nothing about P2's actual determinism guarantee.
"""

import hashlib
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from analysis.detector import Detection
from analysis.ingest import VideoMeta, build_pts_index
from analysis.orchestrator import OfflineConfig, OfflineOrchestrator
from analysis.store import ArtifactStore, ResumeValidationError


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
    fake), so resume and seeding logic can be exercised without loading
    model weights. Output depends only on the per-frame RNG state, which is
    what makes it a faithful stand-in for testing determinism/resume: a
    resumed run reseeds identically to an uninterrupted run at each frame,
    so it must reproduce the same (fake) detections frame-for-frame."""

    crash_after = None

    def __init__(self, **kwargs):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if type(self).crash_after is not None and self.calls > type(self).crash_after:
            raise RuntimeError("simulated crash")
        conf = random.random()
        dx = float(np.random.rand() * 10)
        return [
            Detection(
                track_id=None,
                cls_name="person",
                cls_id=0,
                conf=conf,
                xyxy=(0.0, 0.0, 20.0 + dx, 30.0),
                is_human=True,
                is_threat=False,
            )
        ]


@pytest.fixture(autouse=True)
def _reset_fake_detector():
    FakeDetector.crash_after = None
    yield
    FakeDetector.crash_after = None


def _sidecar_bytes(store: ArtifactStore, pass_name: str) -> tuple[bytes, bytes]:
    frames_path = store.run_dir / "frames" / f"{pass_name}.jsonl"
    dets_path = store.run_dir / "detections" / f"{pass_name}.jsonl"
    frames = frames_path.read_bytes() if frames_path.exists() else b""
    detections = dets_path.read_bytes() if dets_path.exists() else b""
    return frames, detections


class TestSeededReproducibility:
    def test_two_fresh_runs_are_byte_identical(self, monkeypatch):
        monkeypatch.setattr("analysis.detector.Detector", FakeDetector)
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4")
            meta = _make_meta(vpath)
            config = OfflineConfig(seed=123)

            store_a = ArtifactStore(f"{tmp}/out_a", meta.video_hash, "cfg")
            store_a.create()
            OfflineOrchestrator(meta, store_a, config).run_pass_p1()

            store_b = ArtifactStore(f"{tmp}/out_b", meta.video_hash, "cfg")
            store_b.create()
            OfflineOrchestrator(meta, store_b, config).run_pass_p1()

            frames_a, dets_a = _sidecar_bytes(store_a, OfflineOrchestrator.P1_PASS_NAME)
            frames_b, dets_b = _sidecar_bytes(store_b, OfflineOrchestrator.P1_PASS_NAME)
            assert frames_a == frames_b
            assert dets_a == dets_b
            assert dets_a  # sanity: detections were actually produced


class TestInterruptedResume:
    def test_resumed_run_matches_uninterrupted_run(self, monkeypatch):
        monkeypatch.setattr("analysis.detector.Detector", FakeDetector)
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=12)
            meta = _make_meta(vpath)
            config = OfflineConfig(seed=7)
            config_hash = ArtifactStore.config_hash_from_settings(config.to_dict())

            baseline_store = ArtifactStore(f"{tmp}/out_baseline", meta.video_hash, config_hash)
            baseline_store.create()
            OfflineOrchestrator(meta, baseline_store, config).run_pass_p1()
            baseline_frames, baseline_dets = _sidecar_bytes(baseline_store, OfflineOrchestrator.P1_PASS_NAME)

            crashed_store = ArtifactStore(f"{tmp}/out_resumed", meta.video_hash, config_hash)
            crashed_store.create()
            FakeDetector.crash_after = 5
            with pytest.raises(RuntimeError, match="simulated crash"):
                OfflineOrchestrator(meta, crashed_store, config).run_pass_p1()

            FakeDetector.crash_after = None
            resumed_store = ArtifactStore.open_existing(
                f"{tmp}/out_resumed", crashed_store.run_id, meta.video_hash, config_hash
            )
            OfflineOrchestrator(meta, resumed_store, config).run_pass_p1()
            resumed_frames, resumed_dets = _sidecar_bytes(resumed_store, OfflineOrchestrator.P1_PASS_NAME)

            assert resumed_frames == baseline_frames
            assert resumed_dets == baseline_dets

    def test_resume_refuses_on_config_mismatch(self, monkeypatch):
        monkeypatch.setattr("analysis.detector.Detector", FakeDetector)
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=6)
            meta = _make_meta(vpath)
            config_hash = "cfg-original"

            store = ArtifactStore(f"{tmp}/out", meta.video_hash, config_hash)
            store.create()

            with pytest.raises(ResumeValidationError):
                ArtifactStore.open_existing(f"{tmp}/out", store.run_id, meta.video_hash, "cfg-changed")

    def test_resume_with_ignore_region_matches_baseline(self, monkeypatch):
        """Regression guard for the PiP-corner-inset resume bug: P1 filters
        out detections inside ignore_regions before persisting them, and
        (since P1 has no tracker at all) that filtering has no bearing on
        resume correctness — there is no tracker/GMC warm-up state for
        ignored detections to silently omit. A crash-then-resume run must
        still match an uninterrupted baseline byte-for-byte."""
        monkeypatch.setattr("analysis.detector.Detector", FakeDetector)
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=12)
            meta = _make_meta(vpath)
            # FakeDetector always reports a box near the top-left corner;
            # this ignore region covers it, so every detection is dropped.
            config = OfflineConfig(seed=7, ignore_regions=[(0.0, 0.0, 0.5, 0.6)])
            config_hash = ArtifactStore.config_hash_from_settings(config.to_dict())

            baseline_store = ArtifactStore(f"{tmp}/out_baseline", meta.video_hash, config_hash)
            baseline_store.create()
            OfflineOrchestrator(meta, baseline_store, config).run_pass_p1()
            baseline_frames, baseline_dets = _sidecar_bytes(baseline_store, OfflineOrchestrator.P1_PASS_NAME)
            assert baseline_dets == b""  # sanity: the ignore region actually filtered everything

            crashed_store = ArtifactStore(f"{tmp}/out_resumed", meta.video_hash, config_hash)
            crashed_store.create()
            FakeDetector.crash_after = 5
            with pytest.raises(RuntimeError, match="simulated crash"):
                OfflineOrchestrator(meta, crashed_store, config).run_pass_p1()

            FakeDetector.crash_after = None
            resumed_store = ArtifactStore.open_existing(
                f"{tmp}/out_resumed", crashed_store.run_id, meta.video_hash, config_hash
            )
            OfflineOrchestrator(meta, resumed_store, config).run_pass_p1()
            resumed_frames, resumed_dets = _sidecar_bytes(resumed_store, OfflineOrchestrator.P1_PASS_NAME)

            assert resumed_frames == baseline_frames
            assert resumed_dets == baseline_dets

    def test_resume_refuses_on_missing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ResumeValidationError):
                ArtifactStore.open_existing(f"{tmp}/out", "no-such-run", "vhash", "chash")

    def test_resolve_latest_matches_video_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(f"{tmp}/out", "vhash", "chash")
            store.create()

            resolved = ArtifactStore.resolve_latest(f"{tmp}/out", "vhash", "chash")
            assert resolved == store.run_id
            assert ArtifactStore.resolve_latest(f"{tmp}/out", "vhash", "other-chash") is None
            assert ArtifactStore.resolve_latest(f"{tmp}/nonexistent", "vhash", "chash") is None


def _seed_p1_detections(store: ArtifactStore, pass_name: str, n_frames: int) -> None:
    """Seed a store's detections/<pass>.jsonl with a single person moving
    diagonally across the frame, as if a P1 run had already completed —
    used by the P2 determinism tests below, which drive the real BoT-SORT
    tracker directly from persisted detections (no inference involved)."""
    store.record_pass_start(pass_name, {})
    for frame_no in range(n_frames):
        x0 = float(frame_no)
        store.add_detection(
            pass_name,
            frame_no,
            frame_no,
            {
                "xyxy_raw": [x0, 5.0, x0 + 20.0, 40.0],
                "conf": 0.9,
                "cls": "person",
                "cls_id": 0,
                "is_human": True,
                "embedding": None,
            },
        )
        store.add_frame(pass_name, frame_no, {"pts_ms": frame_no * 100.0})
    store.record_pass_complete(pass_name, {"frames_processed": n_frames})


class TestP2TrackingDeterminism:
    """P2 is never checkpointed — it always re-runs in full — so its
    correctness guarantee is that re-running it over the same P1 output
    produces byte-identical tracklets. This must hold against the real
    BoT-SORT tracker, not a fake, since a fake tracker would only prove the
    test harness works, not that P2 itself is deterministic."""

    def test_two_p2_runs_over_same_p1_output_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=10)
            meta = _make_meta(vpath)
            config = OfflineConfig(seed=1)

            store_a = ArtifactStore(f"{tmp}/out_a", meta.video_hash, "cfg")
            store_a.create()
            _seed_p1_detections(store_a, OfflineOrchestrator.P1_PASS_NAME, n_frames=10)
            OfflineOrchestrator(meta, store_a, config).run_pass_p2()

            store_b = ArtifactStore(f"{tmp}/out_b", meta.video_hash, "cfg")
            store_b.create()
            _seed_p1_detections(store_b, OfflineOrchestrator.P1_PASS_NAME, n_frames=10)
            OfflineOrchestrator(meta, store_b, config).run_pass_p2()

            tracklets_a = list(store_a.iter_tracklets(OfflineOrchestrator.P2_PASS_NAME))
            tracklets_b = list(store_b.iter_tracklets(OfflineOrchestrator.P2_PASS_NAME))
            assert tracklets_a == tracklets_b
            assert tracklets_a  # sanity: the tracker actually produced output

    def test_p2_rerun_on_same_store_overwrites_not_appends(self):
        """P2 always fully re-runs; re-running it on a store that already
        has tracklets from a prior invocation must replace, not append to,
        that output (round-3/4's tracker-state-blob problem class doesn't
        apply here, but a truncation bug would silently double every row)."""
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=8)
            meta = _make_meta(vpath)
            config = OfflineConfig(seed=1)

            store = ArtifactStore(f"{tmp}/out", meta.video_hash, "cfg")
            store.create()
            _seed_p1_detections(store, OfflineOrchestrator.P1_PASS_NAME, n_frames=8)

            orch = OfflineOrchestrator(meta, store, config)
            orch.run_pass_p2()
            first_run_rows = list(store.iter_tracklets(OfflineOrchestrator.P2_PASS_NAME))

            orch.run_pass_p2()
            second_run_rows = list(store.iter_tracklets(OfflineOrchestrator.P2_PASS_NAME))

            assert len(second_run_rows) == len(first_run_rows)
            assert second_run_rows == first_run_rows

    def test_p2_does_not_overwrite_p1_raw_detections(self):
        """detections/xyxy_raw must stay P1's pure predict output; tracker-
        adjusted boxes live only in tracklets/."""
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=6)
            meta = _make_meta(vpath)
            config = OfflineConfig(seed=1)

            store = ArtifactStore(f"{tmp}/out", meta.video_hash, "cfg")
            store.create()
            _seed_p1_detections(store, OfflineOrchestrator.P1_PASS_NAME, n_frames=6)
            _, dets_before = _sidecar_bytes(store, OfflineOrchestrator.P1_PASS_NAME)

            OfflineOrchestrator(meta, store, config).run_pass_p2()
            _, dets_after = _sidecar_bytes(store, OfflineOrchestrator.P1_PASS_NAME)

            assert dets_before == dets_after
