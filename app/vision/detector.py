"""YOLO detection + BoT-SORT tracking wrapper.

Model-agnostic: class names are introspected so COCO models (person) and
VisDrone-style models (pedestrian/people) both work, as does any custom
threat model — configure names via HUMAN_CLASSES / THREAT_CLASSES.
Imports torch lazily so unit tests and the web app don't require it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRACKER_YAML = str(Path(__file__).parent / "trackers" / "botsort_drone.yaml")


@dataclass
class Detection:
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
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
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
            for tr in getattr(self.model.predictor, "trackers", []) or []:
                tr.reset()
        except Exception:
            pass  # best effort; tracker re-syncs within a few frames anyway

    def track(self, frame_bgr: np.ndarray) -> list[Detection]:
        res = self.model.track(
            frame_bgr,
            persist=True,
            tracker=TRACKER_YAML,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.wanted,
            device=self.device,
            verbose=False,
        )[0]
        out: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)
        for i in range(len(cls)):
            c = int(cls[i])
            out.append(
                Detection(
                    track_id=None if ids[i] is None else int(ids[i]),
                    cls_name=self.names.get(c, str(c)),
                    conf=float(conf[i]),
                    xyxy=tuple(float(v) for v in xyxy[i]),
                    is_human=c in self.human_ids,
                    is_threat=c in self.threat_ids,
                )
            )
        return out
