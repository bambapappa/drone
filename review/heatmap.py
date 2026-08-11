"""Dwell/coverage heatmap over the persisted tracklet table.

Phase 5 (report §5.9): where people were and for how long — a search-pattern
debrief aid. Cheap by construction: P2's per-(tracklet, frame) rows already
carry every position, so the heatmap is one pass of grid accumulation over a
table the review layer reads anyway. Each row contributes 1/fps seconds of
dwell to the grid cell under the person's foot center ((x0+x1)/2, y1 — the
same anchor point BehaviorAnalyzer and the trail renderer use).

New B29 sidecars carry a local scene foot-point on each P2 row.  Those rows
are accumulated per visually connected scene segment and projected into the
current video frame by the browser.  Older sidecars lack that substrate and
retain the original raw-frame grid as an explicit backward-compatible mode.

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
            "coordinate_space": "raw_frame",
            "segments": [],
        }

    rows = [
        row
        for row in tracklet_rows
        if person_id is None or (person_by_tracklet or {}).get(int(row.get("tracklet_id", -1))) == person_id
    ]
    scene_rows = [
        row for row in rows if row.get("scene_pos") is not None and row.get("scene_segment") is not None
    ]
    if rows and len(scene_rows) == len(rows):
        return _compute_scene_heatmap(scene_rows, fps=fps, grid_w=grid_w, person_id=person_id)

    grid_w = max(1, int(grid_w))
    # Square-ish cells: derive grid_h from the frame aspect ratio.
    grid_h = max(1, round(grid_w * frame_h / frame_w))
    cell_w = frame_w / grid_w
    cell_h = frame_h / grid_h
    dt = 1.0 / fps

    cells = [[0.0] * grid_w for _ in range(grid_h)]
    total = 0.0
    for row in rows:
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
        "coordinate_space": "raw_frame",
        "segments": [],
    }


def _compute_scene_heatmap(
    rows: list[dict[str, Any]],
    fps: float,
    grid_w: int,
    person_id: int | None,
) -> dict[str, Any]:
    """Accumulate sparse scene cells independently per visual segment.

    Segment coordinates are unrelated after GMC loses the scene, so combining
    their extents would create a fictitious shared map.  The client selects the
    segment matching the current video frame and projects only those cells.
    """

    grouped: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        x, y = (float(v) for v in row["scene_pos"])
        grouped.setdefault(int(row["scene_segment"]), []).append((x, y))

    dt = 1.0 / fps
    segments: list[dict[str, Any]] = []
    total = 0.0
    global_max = 0.0
    for segment in sorted(grouped):
        points = grouped[segment]
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        width = max(1, int(grid_w))
        height = max(1, round(width * span_y / span_x))
        cell_w = span_x / width
        cell_h = span_y / height
        counts: dict[tuple[int, int], float] = {}
        for x, y in points:
            gx = min(width - 1, max(0, int((x - min_x) / cell_w)))
            gy = min(height - 1, max(0, int((y - min_y) / cell_h)))
            counts[(gx, gy)] = counts.get((gx, gy), 0.0) + dt

        cells = []
        for (gx, gy), seconds in sorted(counts.items(), key=lambda item: (item[0][1], item[0][0])):
            cells.append(
                {
                    "x": round(min_x + (gx + 0.5) * cell_w, 4),
                    "y": round(min_y + (gy + 0.5) * cell_h, 4),
                    "width": round(cell_w, 4),
                    "height": round(cell_h, 4),
                    "seconds": round(seconds, 3),
                }
            )
            global_max = max(global_max, seconds)
            total += seconds
        segments.append(
            {
                "segment": segment,
                "bounds": [round(min_x, 4), round(min_y, 4), round(max_x, 4), round(max_y, 4)],
                "cells": cells,
            }
        )

    return {
        "coordinate_space": "scene",
        "segments": segments,
        "grid_w": max(1, int(grid_w)),
        "grid_h": None,
        "frame_w": 0,
        "frame_h": 0,
        "cell_s": [],
        "max_s": round(global_max, 3),
        "total_s": round(total, 3),
        "person_id": person_id,
    }
