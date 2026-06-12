"""Video source abstraction: file, RTSP/HTTP URL or camera index.

Files are paced to their native FPS so the rest of the system always sees a
live-like stream — swapping a file for a real drone feed changes nothing
downstream (DECISIONS B9).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".ts", ".mpg", ".mpeg"}


def list_videos(video_dir: str) -> list[dict]:
    d = Path(video_dir)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in VIDEO_EXTS and p.is_file():
            out.append({"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)})
    return out


def is_stream_url(source: str) -> bool:
    return source.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://"))


class VideoSource:
    def __init__(self, source: str, loop: bool = True, max_fps: float = 24.0):
        self.source = source
        self.is_live = is_stream_url(source) or source.isdigit()
        self.loop = loop and not self.is_live
        self.max_fps = max_fps
        self.cap: cv2.VideoCapture | None = None
        self.fps = 25.0
        self.width = 0
        self.height = 0
        self._next_t = 0.0
        self.frame_no = 0
        self.just_looped = False  # set when the file restarted (scene cut)

    def open(self) -> bool:
        src = int(self.source) if self.source.isdigit() else self.source
        self.cap = cv2.VideoCapture(src)
        if self.is_live and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            return False
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1.0 <= fps <= 120.0 else 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next_t = time.monotonic()
        return True

    def read(self):
        """Next frame, paced to wall clock. Returns None at end (non-loop) or error.

        For files: sleeps to native FPS and drops frames if the consumer is
        slow, emulating a live feed. For live URLs: returns frames as they
        arrive. Output rate is additionally capped at max_fps.
        """
        if self.cap is None:
            return None
        step = 1.0 / min(self.fps, self.max_fps) if not self.is_live else 0.0
        skip = max(1, round(self.fps / min(self.fps, self.max_fps))) if not self.is_live else 1

        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_no = 0
                    self.just_looped = True
                    ok, frame = self.cap.read()
                    if not ok:
                        return None
                else:
                    return None
            self.frame_no += 1
            if self.frame_no % skip != 0:
                continue
            if step > 0:
                now = time.monotonic()
                if self._next_t > now:
                    time.sleep(self._next_t - now)
                    self._next_t += step
                else:
                    # Consumer is behind: resync instead of accumulating debt,
                    # and skip ahead if we're more than a frame late.
                    if now - self._next_t > step:
                        self._next_t = now + step
                        continue
                    self._next_t = now + step
            return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
