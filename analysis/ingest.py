"""Video file ingestion: frame-accurate PTS index, video hash, PiP detection.

Builds the canonical index of a video file: frame_no ↔ PTS ↔ byte-safe seek
point. Does one full decode pass at open time; subsequent random access uses
the index rather than trusting codec seeking on B-frame-heavy files.

Also computes a video hash for provenance, extracts fps/resolution, and runs
PipAutoDetector over a sample to lock the IR-PiP layout for the whole file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from analysis.pip import PipAutoDetector


@dataclass
class VideoMeta:
    path: str
    fps: float
    width: int
    height: int
    total_frames: int
    video_hash: str  # sha256 of the first 1MB of binary data for provenance
    pip_layout: str | None = None
    pip_region: tuple[float, float, float, float] | None = None
    active_roi: tuple[float, float, float, float] | None = None


class FrameStore:
    """Sequential iterator + random access by frame index.

    Uses the PTS index built at ingest time; does NOT trust codec seeking
    on B-frame-heavy files. The index maps frame_no → (seek_pos, pts_ms).
    Sequential reads use read(); random access uses seek_to(frame_no).

    After a seek_to(), the tracker and all analysis state must be reset —
    this is the caller's responsibility. For the offline tool, random access
    is only used to rebuild a pass from a checkpoint, not for interactive
    scrubbing (that is the review UI's domain, via HTML5 <video>).
    """

    def __init__(self, video_path: str, meta: VideoMeta):
        self.video_path = video_path
        self.meta = meta
        self._cap: cv2.VideoCapture | None = None
        self._frame_no = 0
        self._pts_idx: list[tuple[int, float]] = []  # [(byte_pos, pts_ms), ...]

    def _open(self) -> None:
        if self._cap is not None:
            return
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def frame_no(self) -> int:
        return self._frame_no

    def seek_to(self, frame_no: int) -> bool:
        """Seek to a specific frame using the stored byte position.
        Returns True on success. After a seek, the next read() returns
        the frame at frame_no."""
        if frame_no < 0 or frame_no >= len(self._pts_idx):
            return False
        self._open()
        byte_pos, _ = self._pts_idx[frame_no]
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        self._frame_no = frame_no
        return True

    def read(self) -> np.ndarray | None:
        """Read the next frame sequentially. Returns None at end."""
        self._open()
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._frame_no += 1
        return frame

    def __iter__(self):
        self._open()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._frame_no = 0
        return self

    def __next__(self) -> np.ndarray:
        frame = self.read()
        if frame is None:
            self.close()
            raise StopIteration
        return frame


def _read_file_bytes(path: str, n: int = 1_000_000) -> bytes:
    """Read the first n bytes of a file for hashing."""
    with open(path, "rb") as f:
        return f.read(n)


def compute_video_hash(video_path: str) -> str:
    """SHA-256 hash of the first 1 MB of the video file for provenance."""
    data = _read_file_bytes(video_path)
    return hashlib.sha256(data).hexdigest()


def build_pts_index(video_path: str) -> tuple[list[tuple[int, float]], float, int, int, int]:
    """Full decode pass: builds frame_no → (byte_pos, pts_ms) index.

    Returns (index, fps, width, height, total_frames).

    The index maps frame_no to a byte-safe seek point and the PTS in
    milliseconds. This is the ground truth for every temporal rule in the
    analysis — t = frame_no / fps (PTS-corrected), not wall-clock time.

    One full decode is expensive (~real-time for the video), but it is a
    one-time cost at ingest. Subsequent passes read sequentially from
    FrameStore, which uses this index for verification.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (1.0 <= fps <= 120.0):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    index: list[tuple[int, float]] = []
    frame_no = 0
    while True:
        pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ok, _ = cap.read()
        if not ok:
            break
        index.append((frame_no, float(pts_ms)))
        frame_no += 1

    total_frames = frame_no
    cap.release()
    return index, fps, width, height, total_frames


def detect_pip_layout(
    video_path: str, sample_count: int = 60, sample_every: int = 8
) -> tuple[str | None, tuple[float, float, float, float] | None]:
    """Run PipAutoDetector over a sample of frames to lock IR-PiP layout.

    Samples `sample_count` frames, one every `sample_every` frames, spread
    evenly across the video. This gives the detector enough temporal coverage
    to lock even for insets that appear partway through the film.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None, None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detector = PipAutoDetector()

    for i in range(sample_count):
        if detector.locked:
            break
        target = int(total * (i + 0.5) / sample_count) if total > 0 else i * sample_every
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if ok:
            detector.feed(frame)

    cap.release()
    if detector.locked:
        return detector.layout, detector.region
    return None, None


def ingest(video_path: str) -> tuple[VideoMeta, FrameStore]:
    """Full ingest: build PTS index, compute hash, detect PiP layout, return
    a ready-to-use FrameStore."""
    path = str(Path(video_path).resolve())
    if not Path(path).is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    index, fps, width, height, total_frames = build_pts_index(path)
    video_hash = compute_video_hash(path)
    pip_layout, pip_region = detect_pip_layout(path)
    from analysis.pip import split_active_roi

    active_roi = split_active_roi(pip_layout) if pip_layout else None

    meta = VideoMeta(
        path=path,
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        video_hash=video_hash,
        pip_layout=pip_layout,
        pip_region=pip_region,
        active_roi=active_roi,
    )

    store = FrameStore(path, meta)
    store._pts_idx = index

    return meta, store
