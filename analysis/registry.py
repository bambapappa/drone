"""Person registry: stable person identities on top of tracker IDs.

The tracker (BoT-SORT) keeps IDs through short occlusions, but a track that
dies (person leaves frame, long occlusion) comes back as a new ID. This
registry re-identifies returning people by appearance (HSV histogram of the
torso region) plus a plausibility check on stabilized position, and assigns
stable person numbers P1, P2, ... The unique-person total is the registry size.

Session-scoped and appearance-based only — no biometrics (see DECISIONS B4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def appearance_hist(frame_bgr: np.ndarray, box_xyxy: tuple[float, float, float, float]) -> np.ndarray | None:
    """HSV histogram of the upper half of the box (torso: most clothing-stable)."""
    x0, y0, x1, y1 = (int(v) for v in box_xyxy)
    H, W = frame_bgr.shape[:2]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 8:
        return None
    roi = frame_bgr[y0 : y0 + max(4, (y1 - y0) // 2), x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    hist = hist.flatten()
    n = np.linalg.norm(hist)
    return hist / n if n > 0 else None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


@dataclass
class Person:
    pid: int
    hist: np.ndarray | None = None
    last_seen: float = 0.0
    first_seen: float = 0.0
    stab_pos: tuple[float, float] = (0.0, 0.0)
    track_ids: set[int] = field(default_factory=set)

    def confirmed(self, min_life_s: float) -> bool:
        return (self.last_seen - self.first_seen) >= min_life_s


class PersonRegistry:
    def __init__(
        self,
        sim_thresh: float = 0.86,
        max_gap_s: float = 60.0,
        max_dist_frac: float = 0.45,
        hist_ema: float = 0.10,
        confirm_s: float = 2.0,
    ):
        self.sim_thresh = sim_thresh
        self.max_gap_s = max_gap_s
        self.max_dist_frac = max_dist_frac
        self.hist_ema = hist_ema
        # Tiny low-confidence detections churn tracker IDs; a person only
        # counts as unique after existing this long (kills count inflation
        # on real footage without hiding boxes).
        self.confirm_s = confirm_s
        self.persons: dict[int, Person] = {}
        self._track_to_pid: dict[int, int] = {}
        self._active_tracks: set[int] = set()
        self._resolved_pids: set[int] = set()
        self._next_pid = 1

    @property
    def unique_total(self) -> int:
        return sum(1 for p in self.persons.values() if p.confirmed(self.confirm_s))

    def begin_frame(self) -> None:
        self._active_tracks = set()
        self._resolved_pids = set()

    def resolve(
        self,
        track_id: int,
        t: float,
        hist: np.ndarray | None,
        stab_pos: tuple[float, float],
        frame_diag: float,
    ) -> int:
        """Map a tracker ID to a stable person ID, re-identifying if possible."""
        self._active_tracks.add(track_id)
        pid = self._track_to_pid.get(track_id)
        if pid is not None and pid in self._resolved_pids:
            # Conflict: another live track already claimed this person in this
            # frame (tracker revived a stale ID after re-ID). The stale mapping
            # loses; re-resolve this track as new/other person.
            del self._track_to_pid[track_id]
            pid = None
        if pid is None:
            pid = self._match_lost(track_id, t, hist, stab_pos, frame_diag)
            if pid is None:
                pid = self._next_pid
                self._next_pid += 1
                self.persons[pid] = Person(pid=pid, first_seen=t)
            self._track_to_pid[track_id] = pid
            self.persons[pid].track_ids.add(track_id)
        self._resolved_pids.add(pid)

        p = self.persons[pid]
        p.last_seen = t
        p.stab_pos = stab_pos
        if hist is not None:
            if p.hist is None:
                p.hist = hist
            else:
                p.hist = (1 - self.hist_ema) * p.hist + self.hist_ema * hist
                n = np.linalg.norm(p.hist)
                if n > 0:
                    p.hist = p.hist / n
        return pid

    def _match_lost(
        self,
        track_id: int,
        t: float,
        hist: np.ndarray | None,
        stab_pos: tuple[float, float],
        frame_diag: float,
    ) -> int | None:
        if hist is None:
            return None
        # Only people whose tracks are all gone are candidates for re-entry.
        active_pids = {self._track_to_pid[tid] for tid in self._active_tracks if tid in self._track_to_pid}
        active_pids |= self._resolved_pids
        best_pid, best_sim = None, self.sim_thresh
        for p in self.persons.values():
            if p.pid in active_pids or p.hist is None:
                continue
            gap = t - p.last_seen
            if gap < 0.3 or gap > self.max_gap_s:
                continue
            dist = float(np.hypot(stab_pos[0] - p.stab_pos[0], stab_pos[1] - p.stab_pos[1]))
            if dist > self.max_dist_frac * frame_diag * (1.0 + gap):
                continue
            sim = cosine(hist, p.hist)
            if sim > best_sim:
                best_pid, best_sim = p.pid, sim
        return best_pid
