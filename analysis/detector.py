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
        self._manual_tracker = None
        if self.tiles > 1:
            # Tiled detections can't go through model.track(); drive BoT-SORT
            # manually. Native (feature-based) ReID is unavailable on merged
            # boxes — the person registry's appearance re-ID still applies.
            from ultralytics.trackers.bot_sort import BOTSORT
            from ultralytics.utils import YAML, IterableSimpleNamespace

            tcfg = YAML.load(self.tracker_yaml)
            tcfg["with_reid"] = False  # no feature ReID on merged tiles; registry handles it
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
            if self._manual_tracker is not None:
                self._manual_tracker.reset()
            for tr in getattr(self.model.predictor, "trackers", []) or []:
                tr.reset()
        except Exception:
            pass  # best effort; tracker re-syncs within a few frames anyway

    def track(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self.tiles > 1:
            return self._track_tiled(frame_bgr)
        res = self.model.track(
            frame_bgr,
            persist=True,
            tracker=self.tracker_yaml,
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

    def _track_tiled(self, frame_bgr: np.ndarray) -> list[Detection]:
        """NxN tiled prediction, global NMS merge, manual BoT-SORT update."""
        import torch
        from ultralytics.engine.results import Boxes

        from analysis.tiling import nms_merge, tile_grid

        h, w = frame_bgr.shape[:2]
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
        det = Boxes(data, orig_shape=(h, w)).cpu().numpy()
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
