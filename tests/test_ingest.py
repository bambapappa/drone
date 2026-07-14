"""Tests for the ingest module: PTS index, video hash, FrameStore.

Uses synthetic videos created at test time so no external test assets are needed.
Tests are ML-free — no model loading.
"""

import tempfile
from pathlib import Path

import cv2
import pytest

from analysis.ingest import (
    FrameStore,
    VideoMeta,
    build_pts_index,
    compute_video_hash,
    detect_pip_layout,
    ingest,
)


def _make_test_video(path: str, n_frames: int = 30, fps: float = 10.0, w: int = 320, h: int = 240) -> str:
    """Create a small synthetic MP4 with moving colored rectangles.
    Uses varied colors so PiP detector does not lock on gray frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    import numpy as np

    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Saturated blue sky + green grass — highly colorful, won't trigger PiP
        frame[0 : h // 2, :] = (200, 100, 40)  # orange-ish sky (BGR)
        frame[h // 2 :, :] = (40, 140, 40)  # green ground
        # Moving red rectangle
        x = int((i / n_frames) * w)
        cv2.rectangle(frame, (x, 80), (x + 30, 130), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    return path


class TestBuildPtsIndex:
    def test_returns_correct_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=30, fps=10.0)
            index, fps, w, h, total = build_pts_index(vpath)
            assert total == 30
            assert w == 320
            assert h == 240
            assert fps == 10.0
            assert len(index) == 30

    def test_index_maps_frame_nos(self):
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=10, fps=10.0)
            index, _, _, _, _ = build_pts_index(vpath)
            # Each entry is (frame_no, pts_ms) — frame_no should be sequential
            for i, (fn, pts) in enumerate(index):
                assert fn == i

    def test_raises_on_missing_file(self):
        with pytest.raises(RuntimeError):
            build_pts_index("/nonexistent/video.mp4")


class TestComputeVideoHash:
    def test_same_file_same_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=5)
            h1 = compute_video_hash(vpath)
            h2 = compute_video_hash(vpath)
            assert h1 == h2
            assert len(h1) == 64  # sha256 hex

    def test_different_files_different_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            v1 = _make_test_video(f"{tmp}/a.mp4", n_frames=5, w=320)
            v2 = _make_test_video(f"{tmp}/b.mp4", n_frames=5, w=640)
            assert compute_video_hash(v1) != compute_video_hash(v2)


class TestFrameStore:
    def test_sequential_iterator(self):
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=20, fps=10.0)
            _, fps, w, h, total = build_pts_index(vpath)
            meta = VideoMeta(
                path=vpath,
                fps=fps,
                width=w,
                height=h,
                total_frames=total,
                video_hash="test",
            )
            store = FrameStore(vpath, meta)
            store._pts_idx = list(enumerate(range(total)))
            frames = list(store)
            assert len(frames) == 20
            for f in frames:
                assert f.shape == (h, w, 3)

    def test_random_access_seek(self):
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=20, fps=10.0)
            _, fps, w, h, total = build_pts_index(vpath)
            meta = VideoMeta(
                path=vpath,
                fps=fps,
                width=w,
                height=h,
                total_frames=total,
                video_hash="test",
            )
            store = FrameStore(vpath, meta)
            store._pts_idx = list(enumerate(range(total)))
            assert store.seek_to(10)
            f = store.read()
            assert f is not None
            assert store.frame_no == 11

    def test_seek_past_end_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=5)
            _, fps, w, h, total = build_pts_index(vpath)
            meta = VideoMeta(
                path=vpath,
                fps=fps,
                width=w,
                height=h,
                total_frames=total,
                video_hash="test",
            )
            store = FrameStore(vpath, meta)
            store._pts_idx = list(enumerate(range(total)))
            assert not store.seek_to(999)


class TestDetectPipLayout:
    def test_no_pip_on_plain_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=30)
            layout, region = detect_pip_layout(vpath, sample_count=10)
            assert layout is None
            assert region is None


class TestIngestIntegration:
    def test_ingest_returns_meta_and_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            vpath = _make_test_video(f"{tmp}/test.mp4", n_frames=20, fps=10.0)
            meta, store = ingest(vpath)
            assert meta.fps == 10.0
            assert meta.total_frames == 20
            assert meta.width == 320
            assert meta.path == str(Path(vpath).resolve())
            assert len(meta.video_hash) == 64
            assert meta.pip_layout is None
            store.close()

    def test_ingest_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ingest("/nonexistent/video.mp4")
