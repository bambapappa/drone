"""Auto-detect an IR / thermal picture-in-picture inset or split-screen half.

Many rescue drones overlay a thermal view as a corner inset or a 50% split.
Run unmasked, that region double-counts people and confuses smoke/fire. The
operator can always set IGNORE_REGIONS / ANALYSIS_ROI manually, but the inset
recurs across real footage, so we detect it automatically when confident.

Signal: a thermal image is essentially grayscale, so the inset is a corner- or
edge-anchored rectangle that is ~100 % low-saturation while the colour aerial
view around it is not. Natural low-saturation areas (asphalt, water, shadow)
top out well below that, giving a clean threshold. A layout is only locked
after agreeing across several frames, so a transient never triggers it.
Validated offline on real footage: 3/3 PiP films detected, 0/5 clean films
misfire (see DECISIONS B20).
"""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

LAYOUTS_CORNER = ("top-right", "top-left", "bottom-right", "bottom-left")


def detect_pip_frame(
    frame_bgr: np.ndarray, work_w: int = 320, sat_thr: int = 40, min_inside: float = 0.985
) -> tuple[str, tuple[float, float, float, float]] | None:
    """One-frame detection. Returns (layout, (x, y, w, h) normalized) or None.

    `min_inside` is the fraction of the candidate region that must be
    low-saturation; 0.985 cleanly separates thermal insets (~1.0) from natural
    low-saturation scenery (≤0.97).
    """
    h, w = frame_bgr.shape[:2]
    sw = work_w
    sh = max(2, int(h * sw / w))
    small = cv2.resize(frame_bgr, (sw, sh))
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1]
    low = (sat < sat_thr).astype(np.uint8)

    probes: list[tuple[str, tuple[int, int, int, int]]] = [
        ("split-right", (sw // 2, 0, sw, sh)),
        ("split-left", (0, 0, sw // 2, sh)),
    ]
    for fw in (0.34, 0.40):
        for fh in (0.42, 0.48):
            cw, ch = int(sw * fw), int(sh * fh)
            probes += [
                ("top-right", (sw - cw, 0, sw, ch)),
                ("top-left", (0, 0, cw, ch)),
                ("bottom-right", (sw - cw, sh - ch, sw, sh)),
                ("bottom-left", (0, sh - ch, cw, sh)),
            ]
    layout = None
    for nm, (x0, y0, x1, y1) in probes:
        if low[y0:y1, x0:x1].mean() >= min_inside:
            layout = nm
            break
    if layout is None:
        return None
    if layout.startswith("split"):
        rect = (0.5, 0.0, 0.5, 1.0) if layout == "split-right" else (0.0, 0.0, 0.5, 1.0)
        return layout, rect
    return layout, _corner_rect(low, layout, sw, sh)


def _corner_rect(low: np.ndarray, layout: str, sw: int, sh: int) -> tuple[float, float, float, float]:
    """Bounding box of the low-saturation component touching the corner,
    clamped to a plausible inset size (falls back to a default if implausible)."""
    corner = {
        "top-right": (sw - 1, 0),
        "top-left": (0, 0),
        "bottom-right": (sw - 1, sh - 1),
        "bottom-left": (0, sh - 1),
    }[layout]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    cl = int(lab[corner[1], corner[0]])
    if cl != 0:
        x, y, bw, bh = stats[cl, :4]
        rw, rh = bw / sw, bh / sh
        if 0.12 <= rw <= 0.55 and 0.12 <= rh <= 0.6:
            return round(x / sw, 3), round(y / sh, 3), round(rw, 3), round(rh, 3)
    return _default_rect(layout)


def _default_rect(layout: str) -> tuple[float, float, float, float]:
    w, h = 0.36, 0.46
    x = 1 - w if "right" in layout else 0.0
    y = 0.0 if "top" in layout else 1 - h
    return round(x, 3), round(y, 3), w, h


class PipAutoDetector:
    """Locks an IR-inset layout once it agrees across `need` of `window` frames.

    Conservative on purpose: if frames disagree or nothing is found, it gives
    up (no region) rather than risk masking real video.
    """

    def __init__(self, window: int = 8, need: int = 3):
        self.window = window
        self.need = need
        self._votes: list[tuple[str, tuple[float, float, float, float]] | None] = []
        self.decided = False
        self.region: tuple[float, float, float, float] | None = None
        self.layout: str | None = None

    def feed(self, frame_bgr: np.ndarray) -> bool:
        """Returns True once a decision (region or definitively none) is made."""
        if self.decided:
            return True
        self._votes.append(detect_pip_frame(frame_bgr))
        if len(self._votes) < self.window:
            return False
        layouts = Counter(v[0] for v in self._votes if v)
        if layouts and layouts.most_common(1)[0][1] >= self.need:
            layout = layouts.most_common(1)[0][0]
            rects = [v[1] for v in self._votes if v and v[0] == layout]
            self.region = tuple(round(float(np.median([r[i] for r in rects])), 3) for i in range(4))
            self.layout = layout
        self.decided = True
        return True

    def reset(self) -> None:
        self._votes.clear()
        self.decided = False
        self.region = None
        self.layout = None
