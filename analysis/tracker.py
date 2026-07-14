"""BoT-SORT tracking pass wrapper — P2.

P2 is a deterministic, always-full re-run: it derives track continuity
purely from P1's already-persisted detections (frame_no, det_id, xyxy_raw,
conf, cls_id) plus a fresh decode of the frames (GMC needs pixels; no
re-inference). It never loads YOLO weights — class ids travel with each
persisted detection record — so it stays cheap.

Because P2 always starts from a fresh Tracker instance and replays every
frame from the start, no state ever needs to be checkpointed: re-running P2
on the same P1 output always reproduces the same tracklets, byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrackedBox:
    det_id: int
    track_id: int
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]


class Tracker:
    def __init__(self, tracker_yaml: str):
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import YAML, IterableSimpleNamespace

        tcfg = YAML.load(tracker_yaml)
        # Appearance ReID is off at the tracker level: P2 associates
        # detections only through frame-to-frame motion continuity
        # (position + GMC). Appearance-based identity across occlusions is
        # the person registry's job (Phase 3), not this pass's.
        tcfg["with_reid"] = False
        self._tracker = BOTSORT(IterableSimpleNamespace(**tcfg))

    def update(self, records: list[dict[str, Any]], frame_bgr: np.ndarray) -> list[TrackedBox]:
        """Advance the tracker by one frame.

        `records` must be P1 detection rows for exactly this frame, in the
        same deterministic order P1 persisted them (by det_id) — BOTSORT
        reports each output row's source index into this array, which is
        how track output is mapped back to det_id. Call every frame,
        including frames with an empty `records` list, so lost tracks age
        out and GMC keeps advancing (never skip empty frames).
        """
        import torch
        from ultralytics.engine.results import Boxes

        h, w = frame_bgr.shape[:2]
        if not records:
            data = torch.zeros((0, 6))
        else:
            data = torch.tensor(
                [[*r["xyxy_raw"], float(r["conf"]), float(r["cls_id"])] for r in records],
                dtype=torch.float32,
            )
        det = Boxes(data, orig_shape=(h, w)).cpu().numpy()
        tracks = self._tracker.update(det, frame_bgr)

        out: list[TrackedBox] = []
        for row in tracks:
            x0, y0, x1, y1, tid, conf, _cls, idx = (float(v) for v in row[:8])
            rec = records[int(idx)]
            out.append(
                TrackedBox(
                    det_id=rec["det_id"],
                    track_id=int(tid),
                    cls_name=rec["cls"],
                    conf=float(conf),
                    xyxy=(x0, y0, x1, y1),
                )
            )
        return out
