"""Tiled inference helpers: NxN overlapping tiles + global NMS merge.

Doubles effective resolution for tiny people at altitude at ~2.4x less
compute than an equivalent single large-image pass (measured, DECISIONS B17).
"""

from __future__ import annotations

import numpy as np


def tile_grid(w: int, h: int, n: int, overlap: float = 0.15) -> list[tuple[int, int, int, int]]:
    """n x n tiles with relative overlap; returns (x0, y0, x1, y1) per tile.

    Edge tiles snap to the frame border so the union covers the whole frame
    (no rounding gaps) and every tile keeps full width/height.
    """
    tw = int(round(w / (n - (n - 1) * overlap)))
    th = int(round(h / (n - (n - 1) * overlap)))
    step_x = (w - tw) / max(n - 1, 1)
    step_y = (h - th) / max(n - 1, 1)
    tiles = []
    for j in range(n):
        y0 = int(round(j * step_y))
        for i in range(n):
            x0 = int(round(i * step_x))
            tiles.append((x0, y0, min(w, x0 + tw), min(h, y0 + th)))
    return tiles


def nms_merge(
    boxes: list[list[float]], scores: list[float], classes: list[int], iou_thresh: float = 0.5
) -> list[int]:
    """Greedy NMS over merged tile detections; returns kept indices.

    Pure numpy so it stays import-light (no torchvision dependency here).
    """
    if not boxes:
        return []
    b = np.asarray(boxes, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    order = s.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix0 = np.maximum(x0[i], x0[rest])
        iy0 = np.maximum(y0[i], y0[rest])
        ix1 = np.minimum(x1[i], x1[rest])
        iy1 = np.minimum(y1[i], y1[rest])
        inter = np.maximum(0.0, ix1 - ix0) * np.maximum(0.0, iy1 - iy0)
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        order = rest[iou <= iou_thresh]
    return keep
