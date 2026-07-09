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


def _drivable_mask(small_bgr: np.ndarray) -> np.ndarray:
    """Rough drivable/open-ground estimate: low-saturation (gray asphalt /
    gravel / concrete / open dirt), non-vegetation, mid-brightness, non-black.
    Trees and grass are green/saturated; the blacked-out IR inset and letterbox
    are near-black. Heuristic — water reads like asphalt by colour, so open
    water can be misclassified (PoC limitation, see DECISIONS B21)."""
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    b, g, r = cv2.split(small_bgr.astype(np.int16))
    vegetation = (g > r + 8) & (g > b + 8)
    openish = (s < 70) & (v > 45) & (v < 235) & ~vegetation
    m = openish.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def _escape_dirs(drivable: np.ndarray, px: int, py: int, w: int, h: int) -> list[str]:
    """Frame-edge directions reachable from (px, py) along a mostly-open
    corridor — i.e. a way out for vehicles."""
    out = []
    for name, (dx, dy) in (("vänster", (-1, 0)), ("höger", (1, 0)), ("uppåt", (0, -1)), ("nedåt", (0, 1))):
        x, y, samples = px, py, []
        while 0 <= x < w and 0 <= y < h:
            samples.append(1 if drivable[y, x] else 0)
            x += dx * 2
            y += dy * 2
        if len(samples) > 5 and float(np.mean(samples)) > 0.65:
            out.append(name)
    return out


class SituationAnalyzer:
    def __init__(
        self,
        min_area: float = 0.004,
        hold_s: float = 2.0,
        flow_ema: float = 0.15,
        base_margin: float = 0.08,
        base_hysteresis: float = 0.15,
        fire_require_smoke: bool = True,
        fire_smoke_radius: float = 0.28,
        fire_smoke_min_frac: float = 0.02,
    ):
        self.min_area = min_area
        self.hold_s = hold_s
        self.flow_ema = flow_ema
        self.base_margin = base_margin
        self.base_hysteresis = base_hysteresis
        # Real fire smokes; a red-tile roof does not. Only accept a fire blob
        # when moving smoke is present in its neighbourhood (DECISIONS B18).
        self.fire_require_smoke = fire_require_smoke
        self.fire_smoke_radius = fire_smoke_radius
        self.fire_smoke_min_frac = fire_smoke_min_frac
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

        fire_blob = _largest_blob(fm)
        if fire_blob is not None and self.fire_require_smoke and not self._smoke_near(sm, fire_blob[0]):
            fire_blob = None  # red roofs / sunset etc. — saturated colour but no smoke
        self.state.fire = self._hold("fire", fire_blob, t, self.state.fire, "_fire_since")
        self.state.smoke = self._hold("smoke", _largest_blob(sm), t, self.state.smoke, "_smoke_since")

        # Smoke drift: median Farneback flow inside the smoke mask, EMA-smoothed.
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape and sm.sum() > 20:
            flow = cv2.calcOpticalFlowFarneback(self._prev_gray, gray, None, 0.5, 2, 11, 2, 5, 1.1, 0)
            ys, xs = np.nonzero(sm)
            v = np.median(flow[ys, xs], axis=0)  # px/frame at WORK_W scale
            v_norm = v / WORK_W  # fraction of frame width per frame
            self._drift = (1 - self.flow_ema) * self._drift + self.flow_ema * v_norm
        self.state.smoke_drift = (float(self._drift[0]), float(self._drift[1]))

        self._suggest_base(danger_norm, small, sm)
        self._prev_gray = gray
        return self.state

    def _smoke_near(self, sm: np.ndarray, fire_pos: tuple[float, float]) -> bool:
        """True if moving smoke occupies a meaningful fraction of the window
        around the fire blob — smoke rises from/around real flames."""
        h, w = sm.shape
        r = int(self.fire_smoke_radius * max(w, h))
        cx, cy = int(fire_pos[0] * w), int(fire_pos[1] * h)
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(w, cx + r), min(h, cy + r)
        window = sm[y0:y1, x0:x1]
        if window.size == 0:
            return False
        return float(window.mean()) >= self.fire_smoke_min_frac

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

    def _suggest_base(
        self, danger_norm: tuple[float, float] | None, small_bgr: np.ndarray, sm: np.ndarray
    ) -> None:
        """Score candidate base locations against the operational criteria:
        away from the danger, not downwind of the smoke, and crucially with an
        escape — either an open corridor reaching a frame edge (drive-through)
        or enough open area to turn rescue vehicles around (DECISIONS B6/B21).
        All in image coordinates; clearly a heuristic suggestion."""
        st = self.state
        h, w = small_bgr.shape[:2]
        drift = st.smoke_drift
        drift_mag = math.hypot(*drift)
        anchor = danger_norm or (st.fire.pos if st.fire else None)

        # A base only makes sense relative to a hazard.
        if anchor is None and drift_mag <= 0.0008:
            st.base, st.base_reasons = None, []
            self._base_target = None
            return

        drivable = _drivable_mask(small_bgr)
        lo, hi = self.base_margin, 1.0 - self.base_margin
        best = None
        for ny in np.linspace(lo, hi, 6):
            for nx in np.linspace(lo, hi, 6):
                cand = self._score_base(float(nx), float(ny), anchor, drift, drift_mag, drivable, sm, w, h)
                if best is None or cand["score"] > best["score"]:
                    best = cand
        target = (best["x"], best["y"])

        reasons: list[str] = []
        if best["exits"]:
            reasons.append(f"Öppen mark åt {', '.join(best['exits'])} — möjlig utväg")
        elif best["turn_room"]:
            reasons.append("Öppen yta nog att vända räddningsfordon på")
        else:
            reasons.append("⚠ Inramad av hinder — kontrollera utväg på plats")
        if drift_mag > 0.0008:
            reasons.append(f"Rök driver åt {_dir_text(*drift)}; bas uppvind/ur rökvägen")
        if anchor is not None:
            reasons.append("Bort från faran (håll säkerhetsavstånd — skala okänd i PoC)")
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

    def _score_base(self, nx, ny, anchor, drift, drift_mag, drivable, sm, w, h) -> dict:
        px, py = int(nx * w), int(ny * h)
        r = max(4, int(0.12 * w))
        x0, y0 = max(0, px - r), max(0, py - r)
        x1, y1 = min(w, px + r), min(h, py + r)
        win = drivable[y0:y1, x0:x1]
        openness = float(win.mean()) if win.size else 0.0
        smoke_pen = float((sm[y0:y1, x0:x1] > 0).mean()) if sm[y0:y1, x0:x1].size else 0.0

        safe_dist = math.hypot(nx - anchor[0], ny - anchor[1]) if anchor else 1.0
        downwind = 0.0
        if anchor is not None and drift_mag > 1e-4:
            dx, dy = nx - anchor[0], ny - anchor[1]
            dn = math.hypot(dx, dy)
            if dn > 1e-3:
                cos = (dx * drift[0] + dy * drift[1]) / (dn * drift_mag)
                downwind = max(0.0, cos)  # pointing the way smoke drifts = exposed

        exits = _escape_dirs(drivable, px, py, w, h)
        turn_room = openness > 0.55
        escape_bonus = 0.6 if exits else (0.3 if turn_room else 0.0)

        score = 1.2 * min(safe_dist, 0.7) + 1.0 * openness + escape_bonus - 1.6 * downwind - 1.3 * smoke_pen
        return {
            "score": score,
            "x": round(nx, 3),
            "y": round(ny, 3),
            "exits": exits,
            "turn_room": turn_room,
            "openness": round(openness, 2),
        }
