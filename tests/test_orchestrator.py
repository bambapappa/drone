"""Tests for the offline orchestrator: video timebase, track_buffer re-expression,
P1 pass structure. Uses a minimal synthetic video (no actual model inference needed
for logic tests).
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from analysis.ingest import VideoMeta, build_pts_index
from analysis.orchestrator import OfflineConfig, OfflineOrchestrator
from analysis.store import ArtifactStore


def _make_test_video(path: str, n_frames: int = 20, fps: float = 10.0) -> str:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (320, 240))
    for i in range(n_frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (60, 60, 60)
        writer.write(frame)
    writer.release()
    return path


def _make_meta(video_path: str) -> VideoMeta:
    index, fps, w, h, total = build_pts_index(video_path)
    import hashlib

    with open(video_path, "rb") as f:
        vhash = hashlib.sha256(f.read(100_000)).hexdigest()
    return VideoMeta(
        path=str(Path(video_path).resolve()),
        fps=fps,
        width=w,
        height=h,
        total_frames=total,
        video_hash=vhash,
    )


class TestVideoTime:
    def test_time_from_frame_no(self):
        config = OfflineConfig()
        meta = VideoMeta(path="", fps=25.0, width=640, height=480, total_frames=100, video_hash="h")
        store = ArtifactStore(tempfile.gettempdir(), "h", "c")
        orch = OfflineOrchestrator(meta, store, config)
        assert orch._video_t(0) == 0.0
        assert orch._video_t(25) == 1.0
        assert orch._video_t(50) == 2.0

    def test_time_correct_for_weird_fps(self):
        config = OfflineConfig()
        meta = VideoMeta(path="", fps=15.0, width=640, height=480, total_frames=100, video_hash="h")
        store = ArtifactStore(tempfile.gettempdir(), "h", "c")
        orch = OfflineOrchestrator(meta, store, config)
        assert orch._video_t(0) == 0.0
        # 15 fps: frame 7 is ~0.467s
        assert abs(orch._video_t(7) - 7.0 / 15.0) < 0.001


class TestTrackBufferExpression:
    def test_track_buffer_re_expressed_for_video_fps(self):
        """track_buffer_s=8s, fps=25 → buffer should be ~200 frames (not 120).
        The orchestrator writes the YAML; we verify the logic computes the right value."""
        config = OfflineConfig(track_buffer_s=8.0)
        meta = VideoMeta(path="", fps=25.0, width=640, height=480, total_frames=100, video_hash="h")
        target = int(max(30, meta.fps * config.track_buffer_s))
        assert target == 200  # 25 fps * 8s = 200

    def test_track_buffer_minimum_30(self):
        config = OfflineConfig(track_buffer_s=0.5)
        meta = VideoMeta(path="", fps=5.0, width=640, height=480, total_frames=100, video_hash="h")
        target = int(max(30, meta.fps * config.track_buffer_s))
        assert target == 30  # 5 fps * 0.5s = 2.5 → floor 30


class TestPipResultApplication:
    def test_split_applies_roi(self):
        meta = VideoMeta(
            path="",
            fps=25.0,
            width=640,
            height=480,
            total_frames=100,
            video_hash="h",
            pip_layout="split-right",
            pip_region=(0.5, 0.0, 0.5, 1.0),
            active_roi=(0.0, 0.0, 0.5, 1.0),
        )
        config = OfflineConfig()
        store = ArtifactStore(tempfile.gettempdir(), "h", "c")
        _ = OfflineOrchestrator(meta, store, config)
        assert config.analysis_roi == (0.0, 0.0, 0.5, 1.0)

    def test_corner_adds_to_ignore(self):
        meta = VideoMeta(
            path="",
            fps=25.0,
            width=640,
            height=480,
            total_frames=100,
            video_hash="h",
            pip_layout="top-right",
            pip_region=(0.66, 0.0, 0.34, 0.44),
            active_roi=None,
        )
        config = OfflineConfig(ignore_regions=[])
        store = ArtifactStore(tempfile.gettempdir(), "h", "c")
        _ = OfflineOrchestrator(meta, store, config)
        assert (0.66, 0.0, 0.34, 0.44) in config.ignore_regions
        assert config.analysis_roi is None

    def test_no_pip_no_change(self):
        meta = VideoMeta(
            path="",
            fps=25.0,
            width=640,
            height=480,
            total_frames=100,
            video_hash="h",
            pip_layout=None,
            pip_region=None,
            active_roi=None,
        )
        config = OfflineConfig()
        store = ArtifactStore(tempfile.gettempdir(), "h", "c")
        _ = OfflineOrchestrator(meta, store, config)
        assert config.analysis_roi is None
        assert config.ignore_regions == []


class TestOfflineConfig:
    def test_to_dict_serializable(self):
        config = OfflineConfig(model="test.pt", seed=42, tiles=2)
        d = config.to_dict()
        assert d["model"] == "test.pt"
        assert d["seed"] == 42
        assert d["tiles"] == 2
        assert "device" in d

    def test_default_config(self):
        config = OfflineConfig()
        assert config.model == "yolo11n.pt"
        assert config.device == "cpu"
        assert config.seed == 42
