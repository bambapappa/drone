"""Situation assessment: fire/smoke heuristics, smoke drift, base suggestion.

Everything here is computed from the actual frames (color masks + optical
flow) — cheap, transparent heuristics clearly labeled as such in the UI.
Coordinates are image-relative; georeferencing needs drone telemetry (PoC 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

WORK_W = 160  # analysis resolution


@dataclass
class Hazard:
    kind: str  # "fire" | "smoke"
    pos: tuple[float, float]  # normalized 0..1
    area: float  # fraction of frame
    since: float


@dataclass
class SituationState:
    fire: Hazard | None = None
    smoke: Hazard | None = None
    smoke_drift: tuple[float, float] = (0.0, 0.0)  # normalized px/s, EMA-smoothed
    base: tuple[float, float] | None = None  # normalized 0..1
    base_reasons: list[str] = field(default_factory=list)


def fire_mask(bgr_small: np.ndarray) -> np.ndarray:
    """Saturated red/orange regions."""
    b, g, r = cv2.split(bgr_small.astype(np.int16))
    m = (r > 150) & (r > g + 30) & (g > b) & (r - b > 70)
    return m.astype(np.uint8)


def smoke_mask(bgr_small: np.ndarray, prev_gray: np.ndarray | None, gray: np.ndarray) -> np.ndarray:
    """Low-saturation gray regions that move (separates smoke from gray asphalt)."""
    hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    static = (s < 60) & (v > 70) & (v < 230)
    if prev_gray is None or prev_gray.shape != gray.shape:
        return np.zeros_like(s, dtype=np.uint8)
    motion = cv2.absdiff(gray, prev_gray) > 6
    m = static & motion
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def _largest_blob(mask: np.ndarray) -> tuple[tuple[float, float], float] | None:
    """((cx, cy) normalized, area fraction) of the largest connected component."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = int(np.argmax(areas)) + 1
    h, w = mask.shape
    cx, cy = centroids[i]
    return (float(cx) / w, float(cy) / h), float(stats[i, cv2.CC_STAT_AREA]) / (w * h)


def _dir_text(dx: float, dy: float) -> str:
    """Direction in screen terms (no compass without telemetry)."""
    ang = math.degrees(math.atan2(dy, dx)) % 360
    names = [
        "höger",
        "höger-ned",
        "nedåt",
        "vänster-ned",
        "vänster",
        "vänster-upp",
        "uppåt",
        "höger-upp",
    ]
    return names[int(((ang + 22.5) % 360) // 45)]


class SituationAnalyzer:
    def __init__(
        self,
        min_area: float = 0.004,
        hold_s: float = 2.0,
        flow_ema: float = 0.15,
        base_margin: float = 0.08,
        base_hysteresis: float = 0.15,
    ):
        self.min_area = min_area
        self.hold_s = hold_s
        self.flow_ema = flow_ema
        self.base_margin = base_margin
        self.base_hysteresis = base_hysteresis
        self.state = SituationState()
        self._prev_gray: np.ndarray | None = None
        self._fire_since: float | None = None
        self._smoke_since: float | None = None
        self._drift = np.zeros(2)
        self._base_target: tuple[float, float] | None = None

    def update(
        self,
        frame_bgr: np.ndarray,
        t: float,
        danger_norm: tuple[float, float] | None,
        ignore: list[tuple[float, float, float, float]] | None = None,
    ) -> SituationState:
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (WORK_W, max(2, int(h * WORK_W / w))))
        if ignore:
            # Black out excluded regions (e.g. IR picture-in-picture): black is
            # outside both the fire and smoke masks' value ranges.
            sh, sw = small.shape[:2]
            for rx, ry, rw, rh in ignore:
                small[int(ry * sh) : int((ry + rh) * sh), int(rx * sw) : int((rx + rw) * sw)] = 0
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        fm = fire_mask(small)
        sm = smoke_mask(small, self._prev_gray, gray)

        self.state.fire = self._hold("fire", _largest_blob(fm), t, self.state.fire, "_fire_since")
        self.state.smoke = self._hold("smoke", _largest_blob(sm), t, self.state.smoke, "_smoke_since")

        # Smoke drift: median Farneback flow inside the smoke mask, EMA-smoothed.
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape and sm.sum() > 20:
            flow = cv2.calcOpticalFlowFarneback(self._prev_gray, gray, None, 0.5, 2, 11, 2, 5, 1.1, 0)
            ys, xs = np.nonzero(sm)
            v = np.median(flow[ys, xs], axis=0)  # px/frame at WORK_W scale
            v_norm = v / WORK_W  # fraction of frame width per frame
            self._drift = (1 - self.flow_ema) * self._drift + self.flow_ema * v_norm
        self.state.smoke_drift = (float(self._drift[0]), float(self._drift[1]))

        self._suggest_base(danger_norm)
        self._prev_gray = gray
        return self.state

    def _hold(self, kind: str, blob, t: float, prev: Hazard | None, since_attr: str) -> Hazard | None:
        """Require min area sustained for hold_s before reporting; keep last
        position briefly to avoid flicker."""
        since = getattr(self, since_attr)
        if blob is not None and blob[1] >= self.min_area:
            if since is None:
                setattr(self, since_attr, t)
                since = t
            if t - since >= self.hold_s:
                return Hazard(kind=kind, pos=blob[0], area=blob[1], since=since)
            return prev
        setattr(self, since_attr, None)
        if prev is not None and t - prev.since < self.hold_s * 4:
            return prev  # short grace to avoid flicker
        return None

    def _suggest_base(self, danger_norm: tuple[float, float] | None) -> None:
        st = self.state
        reasons: list[str] = []
        direction: tuple[float, float] | None = None

        drift_mag = math.hypot(*st.smoke_drift)
        if st.smoke is not None and drift_mag > 0.0008:
            direction = (-st.smoke_drift[0], -st.smoke_drift[1])
            reasons.append(
                f"Rök driver åt {_dir_text(*st.smoke_drift)} — bas uppvind ({_dir_text(*direction)})"
            )
        anchor = danger_norm or (st.fire.pos if st.fire else None)
        if direction is None and anchor is not None:
            direction = (0.5 - anchor[0], 0.5 - anchor[1])
            if math.hypot(*direction) < 0.05:
                direction = (0.0, 1.0)
            reasons.append("Bas på motsatt sida om faran")
        if direction is None:
            st.base, st.base_reasons = None, []
            self._base_target = None
            return

        n = math.hypot(*direction)
        ux, uy = direction[0] / n, direction[1] / n
        start = anchor or (0.5, 0.5)
        lo, hi = self.base_margin, 1.0 - self.base_margin
        # Walk from the anchor toward the frame edge, keep inside margins.
        scale = 1.0
        bx, by = start[0] + ux * scale, start[1] + uy * scale
        bx, by = min(max(bx, lo), hi), min(max(by, lo), hi)
        target = (bx, by)

        if anchor is not None:
            reasons.append("Avstånd till faran: håll säkerhetsavstånd (skala okänd i PoC)")
        reasons.append("Heuristiskt förslag — beslut fattas av räddningsledare")

        # Hysteresis: only move the marker when the target shifts significantly.
        if (
            self._base_target is None
            or math.hypot(target[0] - self._base_target[0], target[1] - self._base_target[1])
            > self.base_hysteresis
        ):
            self._base_target = target
        st.base = self._base_target
        st.base_reasons = reasons
