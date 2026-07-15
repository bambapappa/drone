"""Dwell/coverage heatmap over the persisted tracklet table.

Phase 5 (report §5.9): where people were and for how long — a search-pattern
debrief aid. Cheap by construction: P2's per-(tracklet, frame) rows already
carry every position, so the heatmap is one pass of grid accumulation over a
table the review layer reads anyway. Each row contributes 1/fps seconds of
dwell to the grid cell under the person's foot center ((x0+x1)/2, y1 — the
same anchor point BehaviorAnalyzer and the trail renderer use).

**Coordinate space: raw frame pixels, not the report's "stabilized frame".**
Phase 0 deliberately did not persist per-frame GMC stabilization offsets
(see AGENTS.md's P3 section and DECISIONS B23's "känd approximation" — the
identity gate accepted the same limitation), so raw pixels are the only
substrate on disk. For the 8-film exercise set the drone mostly hovers, so
raw-frame dwell is close to ground dwell; on a fast-moving segment the map
smears along the camera motion. The tool documents this rather than hiding
it; persisting stab offsets in P2 would upgrade this (and P3's gate) in one
place later.

Deterministic: pure function of (tracklet rows, fps, dims, grid size,
person filter). No RNG, no wall-clock.
"""

from __future__ import annotations

from typing import Any, Iterable

DEFAULT_GRID_W = 48


def compute_heatmap(
    tracklet_rows: Iterable[dict[str, Any]],
    fps: float,
    frame_w: int,
    frame_h: int,
    grid_w: int = DEFAULT_GRID_W,
    person_id: int | None = None,
    person_by_tracklet: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Accumulate per-cell dwell seconds over the tracklet table.

    `person_id` + `person_by_tracklet` filter to one person's rows (the
    dossier's per-person map); with person_id None the map covers everyone.
    The mapping passed in should be the *corrected* projection when identity
    corrections exist, so the per-person map follows the same identity the
    rest of the UI shows.

    Boxes can extend past the frame edge (the tracker's Kalman prediction
    isn't clipped), so cell indices are clamped rather than dropped — dwell
    at the edge is still dwell.
    """
    if fps <= 0 or frame_w <= 0 or frame_h <= 0:
        return {
            "grid_w": 0,
            "grid_h": 0,
            "frame_w": frame_w,
            "frame_h": frame_h,
            "cell_s": [],
            "max_s": 0.0,
            "total_s": 0.0,
            "person_id": person_id,
        }

    grid_w = max(1, int(grid_w))
    # Square-ish cells: derive grid_h from the frame aspect ratio.
    grid_h = max(1, round(grid_w * frame_h / frame_w))
    cell_w = frame_w / grid_w
    cell_h = frame_h / grid_h
    dt = 1.0 / fps

    cells = [[0.0] * grid_w for _ in range(grid_h)]
    total = 0.0
    pmap = person_by_tracklet or {}
    for row in tracklet_rows:
        if person_id is not None and pmap.get(int(row.get("tracklet_id", -1))) != person_id:
            continue
        x0, y0, x1, y1 = (float(v) for v in row["xyxy"])
        foot_x = (x0 + x1) / 2.0
        foot_y = y1
        gx = min(grid_w - 1, max(0, int(foot_x / cell_w)))
        gy = min(grid_h - 1, max(0, int(foot_y / cell_h)))
        cells[gy][gx] += dt
        total += dt

    max_s = max((v for r in cells for v in r), default=0.0)
    return {
        "grid_w": grid_w,
        "grid_h": grid_h,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "cell_s": [[round(v, 3) for v in r] for r in cells],
        "max_s": round(max_s, 3),
        "total_s": round(total, 3),
        "person_id": person_id,
    }
