"""Acceptance tests for explicit --resume and seeded reproducibility.

Uses a fake, dependency-free Detector (no real YOLO/torch) so these tests run
without ML weights, while still exercising the real orchestrator/store
checkpoint-resume and per-frame seeding machinery end to end.
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
    for _ in range(n_frames):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:] = (60, 90, 120)
        cv2.rectangle(frame, (5, 5), (40, 50), (0, 0, 255), -1)
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
    """Deterministic-if-seeded stand-in for the real YOLO Detector, so resume
    and seeding logic can be exercised without loading model weights.

    Stateful like the real BOT-SORT tracker: track_id and a fake GMC-style
    drift offset both accumulate across calls on one instance. A fresh
    instance therefore produces different output than one warmed up over
    prior frames — the same divergence that made the original resume bug
    invisible to a stateless fake."""

    calls = 0
    crash_after = None

    def __init__(self, **kwargs):
        self._next_track_id = 1
        self._gmc_offset = 0.0

    def track(self, frame):
        type(self).calls += 1
        if type(self).crash_after is not None and type(self).calls > type(self).crash_after:
            raise RuntimeError("simulated crash")
        conf = random.random()
        dx = float(np.random.rand() * 10)
        self._gmc_offset += 1.0
        track_id = self._next_track_id
        self._next_track_id += 1
        return [
            Detection(
                track_id=track_id,
                cls_name="person",
                conf=conf,
                xyxy=(self._gmc_offset, 0.0, 20.0 + dx + self._gmc_offset, 30.0),
                is_human=True,
                is_threat=False,
            )
        ]

    def get_tracker_state(self):
        return {"next_track_id": self._next_track_id, "gmc_offset": self._gmc_offset}

    def restore_tracker_state(self, state):
        self._next_track_id = state["next_track_id"]
        self._gmc_offset = state["gmc_offset"]


@pytest.fixture(autouse=True)
def _reset_fake_detector():
    FakeDetector.calls = 0
    FakeDetector.crash_after = None
    yield
    FakeDetector.calls = 0
    FakeDetector.crash_after = None


def _sidecar_bytes(store: ArtifactStore, pass_name: str) -> tuple[bytes, bytes]:
    frames = (store.run_dir / "frames" / f"{pass_name}.jsonl").read_bytes()
    detections = (store.run_dir / "detections" / f"{pass_name}.jsonl").read_bytes()
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

            frames_a, dets_a = _sidecar_bytes(store_a, "p1_detect")
            frames_b, dets_b = _sidecar_bytes(store_b, "p1_detect")
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
            baseline_frames, baseline_dets = _sidecar_bytes(baseline_store, "p1_detect")

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
            resumed_frames, resumed_dets = _sidecar_bytes(resumed_store, "p1_detect")

            assert resumed_frames == baseline_frames
            assert resumed_dets == baseline_dets

    def test_resume_refuses_without_tracker_state_blob(self, monkeypatch):
        """A crash before the first checkpoint interval leaves frame/detection
        rows on disk but no tracker/GMC state blob yet. Resuming from there
        must be refused outright, never silently continued with a cold
        tracker — that would produce a resumed run that diverges from an
        uninterrupted one, exactly the bug the stateful FakeDetector above is
        designed to catch."""
        monkeypatch.setattr("analysis.detector.Detector", FakeDetector)
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=12)
            meta = _make_meta(vpath)
            config = OfflineConfig(seed=7)
            config_hash = ArtifactStore.config_hash_from_settings(config.to_dict())

            crashed_store = ArtifactStore(f"{tmp}/out", meta.video_hash, config_hash)
            crashed_store.create()
            FakeDetector.crash_after = 5
            with pytest.raises(RuntimeError, match="simulated crash"):
                OfflineOrchestrator(meta, crashed_store, config).run_pass_p1()
            FakeDetector.crash_after = None

            assert not (crashed_store.run_dir / "checkpoints" / "p1_detect" / "tracker_state.pkl").exists()

            resumed_store = ArtifactStore.open_existing(
                f"{tmp}/out", crashed_store.run_id, meta.video_hash, config_hash
            )
            with pytest.raises(ResumeValidationError):
                OfflineOrchestrator(meta, resumed_store, config).run_pass_p1()

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
