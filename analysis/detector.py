"""YOLO detection wrapper — P1's stateless detection pass.

Model-agnostic: class names are introspected so COCO models (person) and
VisDrone-style models (pedestrian/people) both work, as does any custom
threat model — configure names via HUMAN_CLASSES / THREAT_CLASSES.
Imports torch lazily so unit tests and the web app don't require it.

Detection (this module) and tracking (analysis.tracker) are separate passes:
P1 runs tiled inference and persists raw detections only, with no tracker
involved; P2 (analysis.tracker.Tracker) re-derives track continuity purely
from those persisted detections. This is what makes P1 trivially resumable
(nothing stateful crosses its checkpoint boundary) and P2 a cheap, always-
full re-run (deterministic given the same persisted detections).
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
    # None for every P1 detection; P2 (analysis.tracker.Tracker) assigns it.
    track_id: int | None
    cls_name: str
    cls_id: int
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
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.tiles = max(1, tiles)
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

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Stateless per-frame detection: tiled inference + NMS merge.

        Returns raw predict boxes only — never tracker-adjusted, since P1
        never touches a tracker. track_id is always None here.
        """
        h, w = frame_bgr.shape[:2]
        boxes = self._predict_boxes(frame_bgr, h, w)
        out: list[Detection] = []
        for x0, y0, x1, y1, conf, cls in zip(
            boxes.xyxy[:, 0], boxes.xyxy[:, 1], boxes.xyxy[:, 2], boxes.xyxy[:, 3], boxes.conf, boxes.cls
        ):
            c = int(cls)
            out.append(
                Detection(
                    track_id=None,
                    cls_name=self.names.get(c, str(c)),
                    cls_id=c,
                    conf=float(conf),
                    xyxy=(float(x0), float(y0), float(x1), float(y1)),
                    is_human=c in self.human_ids,
                    is_threat=c in self.threat_ids,
                )
            )
        return out

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
            data = torch.zeros((0, 6))
        else:
            data = torch.tensor(
                [[*boxes[i], scores[i], float(classes[i])] for i in keep], dtype=torch.float32
            )
        return Boxes(data, orig_shape=(h, w)).cpu().numpy()
