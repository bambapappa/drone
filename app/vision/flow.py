"""Optical-flow helpers and display-box smoothing.

The render thread runs faster than detection. Between detections each box is
carried along by local optical flow; flow motion is fed forward 1:1 into the
display filter while detection corrections are absorbed as slew-limited
glides, so boxes neither jitter, lag nor teleport (DECISIONS.md B3).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)


class OneEuroFilter:
    """Scalar One Euro filter (Casiez et al. 2012).

    Low cutoff at rest removes jitter; the beta term raises the cutoff with
    speed so fast motion is tracked with minimal lag.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._t is None or self._x is None:
            self._x, self._t = x, t
            return x
        dt = t - self._t
        if dt <= 0:
            return self._x
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self._x) / dt
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        self._t = t
        return self._x

    def reset_to(self, x: float, t: float) -> None:
        self._x, self._t, self._dx = x, t, 0.0


class BoxFilter:
    """Display smoother for a box (cx, cy, w, h), called every render frame.

    dt-aware EMA toward the raw box plus a slew limit: the box may move at
    most `slew` box-dimensions per second toward the target. Smooth pursuit
    passes through almost unfiltered (small residuals), while detection
    corrections turn into short glides instead of teleports — regardless of
    how large the correction is. Deterministic and frame-rate independent.
    """

    def __init__(self, tau_pos: float = 0.12, tau_size: float = 0.18, slew: float = 3.0):
        self.tau_pos = tau_pos
        self.tau_size = tau_size
        self.slew = slew
        self._box: list[float] | None = None
        self._t: float | None = None

    def __call__(
        self,
        box: tuple[float, float, float, float],
        t: float,
        ff: tuple[float, float] = (0.0, 0.0),
    ):
        """`ff` is a feed-forward displacement (the optical-flow motion this
        frame): applied 1:1 so camera/scene motion never lags, while the
        EMA+slew only works on the remaining correction residual."""
        if self._box is None or self._t is None:
            self._box, self._t = list(box), t
            return tuple(box)
        dt = t - self._t
        if dt <= 0:
            return tuple(self._box)
        self._t = t
        self._box[0] += ff[0]
        self._box[1] += ff[1]
        a_pos = 1.0 - math.exp(-dt / self.tau_pos)
        a_size = 1.0 - math.exp(-dt / self.tau_size)
        dim = max(self._box[2], self._box[3], 8.0)
        max_step = self.slew * dim * dt
        for i, a in enumerate((a_pos, a_pos, a_size, a_size)):
            step = a * (box[i] - self._box[i])
            if step > max_step:
                step = max_step
            elif step < -max_step:
                step = -max_step
            self._box[i] += step
        return tuple(self._box)

    def reset_to(self, box: tuple[float, float, float, float], t: float) -> None:
        """Re-seed the state (track handover/re-acquire: start gliding from
        here instead of teleporting)."""
        self._box, self._t = list(box), t


class GlobalMotion:
    """Per-frame camera translation estimate from sparse LK flow.

    Works on an already-downscaled gray image; `scale` converts back to
    source pixels (source_px = small_px * scale). `offset` accumulates the
    scene shift in source pixels so positions can be expressed in a
    camera-stabilized frame: stab = screen - offset.
    """

    def __init__(self):
        self._prev: np.ndarray | None = None
        self.last_shift = (0.0, 0.0)
        self.offset = np.zeros(2, dtype=np.float64)

    def update(self, small_gray: np.ndarray, scale: float) -> tuple[float, float]:
        # Implausibly large shifts (scene cuts, decode glitches) are ignored
        # rather than poisoning the accumulated offset.
        max_shift = 0.25 * small_gray.shape[1]
        shift = (0.0, 0.0)
        if self._prev is not None and self._prev.shape == small_gray.shape:
            pts = cv2.goodFeaturesToTrack(self._prev, maxCorners=120, qualityLevel=0.01, minDistance=12)
            if pts is not None and len(pts) >= 8:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(self._prev, small_gray, pts, None, **LK_PARAMS)
                if nxt is not None:
                    good = st.reshape(-1) == 1
                    if good.sum() >= 8:
                        d = (nxt.reshape(-1, 2) - pts.reshape(-1, 2))[good]
                        med = np.median(d, axis=0)
                        if abs(med[0]) < max_shift and abs(med[1]) < max_shift:
                            shift = (float(med[0]) * scale, float(med[1]) * scale)
        self._prev = small_gray
        self.last_shift = shift
        self.offset += np.asarray(shift)
        return shift

    def reset_motion(self) -> None:
        """Drop the previous frame (e.g. after a scene cut) so the next
        update measures nothing instead of garbage."""
        self._prev = None

    def to_stab(self, x: float, y: float) -> tuple[float, float]:
        return x - float(self.offset[0]), y - float(self.offset[1])

    def to_screen(self, sx: float, sy: float) -> tuple[float, float]:
        return sx + float(self.offset[0]), sy + float(self.offset[1])


def local_box_flow(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    box: tuple[float, float, float, float],
    fallback: tuple[float, float],
) -> tuple[float, float]:
    """Median LK flow of points inside `box` (cx, cy, w, h in pixels).

    Falls back to the global camera shift when too few points track, so a
    featureless box still moves with the scene instead of freezing.
    """
    cx, cy, w, h = box
    H, W = prev_gray.shape[:2]
    x0 = max(0, int(cx - w / 2))
    y0 = max(0, int(cy - h / 2))
    x1 = min(W, int(cx + w / 2))
    y1 = min(H, int(cy + h / 2))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return fallback
    roi = prev_gray[y0:y1, x0:x1]
    pts = cv2.goodFeaturesToTrack(roi, maxCorners=24, qualityLevel=0.05, minDistance=5)
    if pts is None or len(pts) < 4:
        return fallback
    pts = pts.reshape(-1, 2) + np.array([x0, y0], dtype=np.float32)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts.reshape(-1, 1, 2), None, **LK_PARAMS)
    if nxt is None:
        return fallback
    good = st.reshape(-1) == 1
    if good.sum() < 4:
        return fallback
    d = (nxt.reshape(-1, 2) - pts)[good]
    med = np.median(d, axis=0)
    return float(med[0]), float(med[1])
