"""Behavior analysis in camera-stabilized coordinates.

Classifies each tracked person as:
  ok            — normal movement
  still         — no significant motion for a sustained period (worst case:
                  unconscious; aspect ratio > prone threshold marks lying down)
  toward_danger — sustained movement toward the marked danger point

Speeds are normalized by the person's own box height ("body heights per
second") which keeps thresholds roughly scale-invariant across altitudes.
All thresholds are fixed config values: same config => same behavior on
footage the system has never seen (DECISIONS B5, B12).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

STATUS_OK = "ok"
STATUS_STILL = "still"
STATUS_TOWARD = "toward_danger"


@dataclass
class BehaviorConfig:
    window_s: float = 6.0
    min_history_s: float = 3.0
    still_speed: float = 0.10
    still_time_s: float = 4.0
    toward_speed: float = 0.25
    toward_angle_deg: float = 40.0
    toward_time_s: float = 1.5
    prone_aspect: float = 1.4


@dataclass
class _State:
    # (t, stab_x, stab_y, box_h, aspect)
    hist: deque = field(default_factory=lambda: deque(maxlen=512))
    still_since: float | None = None
    toward_since: float | None = None
    status: str = STATUS_OK
    prone: bool = False
    speed: float = 0.0


class BehaviorAnalyzer:
    def __init__(self, cfg: BehaviorConfig | None = None):
        self.cfg = cfg or BehaviorConfig()
        self._states: dict[int, _State] = {}

    def update(
        self,
        pid: int,
        t: float,
        stab_pos: tuple[float, float],
        box_h: float,
        aspect: float,
        danger_stab: tuple[float, float] | None,
    ) -> tuple[str, bool, float]:
        """Feed one observation; returns (status, prone, norm_speed)."""
        st = self._states.setdefault(pid, _State())
        st.hist.append((t, stab_pos[0], stab_pos[1], max(box_h, 1.0), aspect))
        self._trim(st, t)

        span = t - st.hist[0][0]
        if span < self.cfg.min_history_s:
            st.status, st.prone = STATUS_OK, False
            return st.status, st.prone, st.speed

        speed, direction = self._motion(st)
        st.speed = speed
        st.prone = self._median_aspect(st) > self.cfg.prone_aspect

        # --- still ---
        if speed < self.cfg.still_speed:
            if st.still_since is None:
                st.still_since = t
        else:
            st.still_since = None

        # --- toward danger ---
        toward = False
        if danger_stab is not None and speed > self.cfg.toward_speed and direction is not None:
            last = st.hist[-1]
            dx, dy = danger_stab[0] - last[1], danger_stab[1] - last[2]
            dist = math.hypot(dx, dy)
            if dist > 1.0:
                cos_a = (direction[0] * dx + direction[1] * dy) / dist
                toward = cos_a > math.cos(math.radians(self.cfg.toward_angle_deg))
        if toward:
            if st.toward_since is None:
                st.toward_since = t
        else:
            st.toward_since = None

        if st.still_since is not None and t - st.still_since >= self.cfg.still_time_s:
            st.status = STATUS_STILL
        elif st.toward_since is not None and t - st.toward_since >= self.cfg.toward_time_s:
            st.status = STATUS_TOWARD
        else:
            st.status = STATUS_OK
        return st.status, st.prone, st.speed

    def status_of(self, pid: int) -> str:
        st = self._states.get(pid)
        return st.status if st else STATUS_OK

    def drop_inactive(self, active_pids: set[int]) -> None:
        for pid in list(self._states):
            if pid not in active_pids:
                del self._states[pid]

    def _trim(self, st: _State, t: float) -> None:
        while st.hist and t - st.hist[0][0] > self.cfg.window_s:
            st.hist.popleft()

    def _motion(self, st: _State) -> tuple[float, tuple[float, float] | None]:
        """Normalized speed (body heights / s) and mean unit direction over the
        recent half of the window — robust to single-frame jitter."""
        h = list(st.hist)
        if len(h) < 4:
            return 0.0, None
        mid = h[len(h) // 2]
        last = h[-1]
        dt = last[0] - mid[0]
        if dt < 0.5:
            mid = h[0]
            dt = last[0] - mid[0]
            if dt <= 0:
                return 0.0, None
        dx, dy = last[1] - mid[1], last[2] - mid[2]
        body = self._median_h(h)
        speed = math.hypot(dx, dy) / dt / body
        norm = math.hypot(dx, dy)
        direction = (dx / norm, dy / norm) if norm > 1e-6 else None
        return speed, direction

    @staticmethod
    def _median_h(hist: list) -> float:
        hs = sorted(p[3] for p in hist)
        return max(hs[len(hs) // 2], 1.0)

    @staticmethod
    def _median_aspect(st: _State) -> float:
        a = sorted(p[4] for p in st.hist)
        return a[len(a) // 2] if a else 0.0
