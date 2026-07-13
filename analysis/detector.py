"""YOLO detection + BoT-SORT tracking wrapper.

Model-agnostic: class names are introspected so COCO models (person) and
VisDrone-style models (pedestrian/people) both work, as does any custom
threat model — configure names via HUMAN_CLASSES / THREAT_CLASSES.
Imports torch lazily so unit tests and the web app don't require it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TRACKER_YAML = str(Path(__file__).parent / "trackers" / "botsort_drone.yaml")


@dataclass
class Detection:
    # P1/P2-internal tracking lineage, not a stable public identity — that is
    # person_id, assigned by the Phase 3 registry across occlusions/re-entries.
    track_id: int | None
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]
    is_human: bool
    is_threat: bool


class Detector:
    def __init__(
        self,
        model_path: str,
        device: str,
        imgsz: int,
        conf: float,
        iou: float,
        human_classes: set[str],
        threat_classes: set[str],
        tiles: int = 1,
        tracker_yaml: str | None = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.tiles = max(1, tiles)
        self.tracker_yaml = tracker_yaml or TRACKER_YAML
        # Detection (model.predict) and tracking (manual BOTSORT.update) are
        # always split, for tiles==1 as much as tiles>1: this is what makes
        # tracker/GMC state replay-able on resume without re-running
        # inference (step_tracker below). Native (feature-based) ReID is
        # unavailable on this manually-driven path — the person registry's
        # appearance re-ID handles it instead.
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import YAML, IterableSimpleNamespace

        tcfg = YAML.load(self.tracker_yaml)
        tcfg["with_reid"] = False
        self._manual_tracker = BOTSORT(IterableSimpleNamespace(**tcfg))
        self.names: dict[int, str] = dict(self.model.names)
        lower = {i: n.lower() for i, n in self.names.items()}
        self.human_ids = {i for i, n in lower.items() if n in human_classes}
        self.threat_ids = {i for i, n in lower.items() if n in threat_classes}
        self.wanted = sorted(self.human_ids | self.threat_ids)
        if not self.human_ids:
            raise ValueError(
                f"Model {model_path} has no class matching {human_classes}; "
                f"available: {sorted(lower.values())}"
            )

    def reset_tracker(self) -> None:
        """Reset BoT-SORT state after a scene cut (file loop, source glitch)."""
        try:
            self._manual_tracker.reset()
        except Exception:
            pass  # best effort; tracker re-syncs within a few frames anyway

    def track(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run inference then advance the tracker — the normal per-frame path."""
        h, w = frame_bgr.shape[:2]
        det = self._predict_boxes(frame_bgr, h, w)
        return self.step_tracker(det, frame_bgr)

    def step_tracker(self, det: Any, frame_bgr: np.ndarray) -> list[Detection]:
        """Advance ONLY the tracker on a prebuilt detections array (no inference).

        The replay primitive: on resume, feeding already-persisted detections
        through this call (for frames [0, resume_point)) reproduces the same
        tracker.update() sequence an uninterrupted run performed, so
        tracker/GMC state at the resume point matches by construction —
        without re-running the expensive inference step.
        """
        tracks = self._manual_tracker.update(det, frame_bgr)
        out: list[Detection] = []
        for row in tracks:
            x0, y0, x1, y1, tid, conf, cls = (float(v) for v in row[:7])
            c = int(cls)
            out.append(
                Detection(
                    track_id=int(tid),
                    cls_name=self.names.get(c, str(c)),
                    conf=conf,
                    xyxy=(x0, y0, x1, y1),
                    is_human=c in self.human_ids,
                    is_threat=c in self.threat_ids,
                )
            )
        return out

    def build_replay_boxes(self, records: list[dict[str, Any]], shape: tuple[int, int]) -> Any:
        """Build a detections array from persisted detection records
        (xyxy_raw + conf + cls) in the same format `_predict_boxes` produces,
        for feeding into `step_tracker` during resume replay warm-up."""
        import torch
        from ultralytics.engine.results import Boxes

        if not records:
            data = torch.zeros((0, 6))
        else:
            rows = []
            for r in records:
                x0, y0, x1, y1 = r["xyxy_raw"]
                rows.append([x0, y0, x1, y1, float(r["conf"]), float(self._cls_id(r["cls"]))])
            data = torch.tensor(rows, dtype=torch.float32)
        return Boxes(data, orig_shape=shape).cpu().numpy()

    def _cls_id(self, cls_name: str) -> int:
        for i, n in self.names.items():
            if n == cls_name:
                return i
        return -1

    def _predict_boxes(self, frame_bgr: np.ndarray, h: int, w: int) -> Any:
        """NxN tiled prediction + global NMS merge (n=1 is a single full-frame tile)."""
        import torch
        from ultralytics.engine.results import Boxes

        from analysis.tiling import nms_merge, tile_grid

        boxes: list[list[float]] = []
        scores: list[float] = []
        classes: list[int] = []
        for x0, y0, x1, y1 in tile_grid(w, h, self.tiles):
            res = self.model.predict(
                frame_bgr[y0:y1, x0:x1],
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                classes=self.wanted,
                device=self.device,
                verbose=False,
            )[0]
            if res.boxes is None:
                continue
            for b in res.boxes:
                bx0, by0, bx1, by1 = b.xyxy[0].tolist()
                boxes.append([bx0 + x0, by0 + y0, bx1 + x0, by1 + y0])
                scores.append(float(b.conf[0]))
                classes.append(int(b.cls[0]))

        keep = nms_merge(boxes, scores, classes)
        if not keep:
            # Still step the tracker (ages out lost tracks, advances GMC).
            data = torch.zeros((0, 6))
        else:
            data = torch.tensor(
                [[*boxes[i], scores[i], float(classes[i])] for i in keep], dtype=torch.float32
            )
        return Boxes(data, orig_shape=(h, w)).cpu().numpy()
