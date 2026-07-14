"""Sequential, single-process, multi-pass analysis orchestrator.

Replaces the two-thread realtime Pipeline/PipelineManager for the offline tool.
Passes are sequential and each pass declares what it reads and writes against
the artifact schema. Adding a new detection capability = adding a pass, not
touching existing ones.

Phase 0 wires P1 (stateless detection) and P2 (tracking) end to end.
Later phases add P3 (identity), P4 (behavior), and P5 (event derivation) —
each as a separate pass that reads from the artifact store.

P1/P2 split: P1 runs tiled inference and persists raw detections only —
no tracker involved, so nothing stateful crosses its checkpoint boundary and
it is trivially, bit-identically resumable. P2 re-derives track continuity
purely from P1's persisted detections plus a fresh decode (GMC needs
pixels, not re-inference); it is never checkpointed and always re-runs in
full, which is cheap and deterministic given the same P1 output.
"""

from __future__ import annotations

import hashlib
import os
import random
import time as _time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from analysis.ingest import FrameStore, VideoMeta
from analysis.registry import appearance_hist
from analysis.store import ArtifactStore

FLOW_W = 480


def _seed_rng(seed: int) -> None:
    """Seed every RNG the analysis path touches. Called once at run start and
    again (with a derived seed) before every frame, so a resumed run's RNG
    state matches an uninterrupted run frame-for-frame."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed % (2**63))
    except ImportError:
        pass


def _frame_seed(base_seed: int, pass_name: str, frame_no: int) -> int:
    """Derive a per-frame seed from (base_seed, pass, frame_no).

    Stateless and independent of prior frames — a resumed run reseeds each
    frame identically to an uninterrupted run, with no RNG state to persist
    across a checkpoint. Uses a stable hash (not Python's builtin hash(),
    which is randomly salted per-process unless PYTHONHASHSEED is fixed) so
    the same triple always maps to the same seed across separate invocations.
    """
    payload = f"{base_seed}:{pass_name}:{frame_no}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


@dataclass
class OfflineConfig:
    """Settings for an offline analysis run. Mirrors the realtime Settings
    fields that are relevant, plus offline-only fields."""

    model: str = "yolo11n.pt"
    device: str = "cpu"
    imgsz: int = 640
    conf: float = 0.30
    iou: float = 0.50
    tiles: int = 1
    human_classes: set[str] = field(default_factory=lambda: {"person", "pedestrian", "people"})
    threat_classes: set[str] = field(default_factory=set)

    # Analysis ROI and ignore regions (normalized)
    analysis_roi: tuple[float, float, float, float] | None = None
    ignore_regions: list[tuple[float, float, float, float]] = field(default_factory=list)
    pip_autodetect: bool = True

    # Behavior thresholds
    beh_window_s: float = 6.0
    beh_min_history_s: float = 3.0
    beh_still_speed: float = 0.10
    beh_still_time_s: float = 4.0
    beh_toward_speed: float = 0.25
    beh_toward_angle_deg: float = 40.0
    beh_toward_time_s: float = 1.5
    beh_prone_aspect: float = 1.4

    # Re-ID registry
    reid_sim_thresh: float = 0.86
    reid_max_gap_s: float = 60.0
    reid_max_dist_frac: float = 0.45

    # Situation assessment
    hazard_min_area: float = 0.004
    hazard_hold_s: float = 2.0
    smoke_flow_ema: float = 0.15
    base_margin: float = 0.08
    base_hysteresis: float = 0.15
    fire_require_smoke: bool = True

    # Determinism
    seed: int = 42

    # Track buffer re-expression: seconds (not frames)
    # track_buffer in yaml = fps * track_buffer_s
    track_buffer_s: float = 8.0

    def to_dict(self) -> dict[str, Any]:
        """Serializable config for the manifest."""
        return {
            "model": self.model,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "tiles": self.tiles,
            "human_classes": sorted(self.human_classes),
            "pip_autodetect": self.pip_autodetect,
            "seed": self.seed,
            "track_buffer_s": self.track_buffer_s,
        }


class OfflineOrchestrator:
    """Sequential multi-pass analysis driver.

    Usage:
        orchestrator = OfflineOrchestrator(meta, store, config)
        orchestrator.run_pass_p1()  # detection pass (checkpointed/resumable)
        orchestrator.run_pass_p2()  # tracking pass (always a full re-run)
        # Later phases:
        # orchestrator.run_pass_p3()  # identity
        # orchestrator.run_pass_p4()  # behavior
        # orchestrator.run_pass_p5()  # event derivation
    """

    P1_PASS_NAME = "p1_detect"
    P2_PASS_NAME = "p2_track"

    def __init__(self, meta: VideoMeta, store: ArtifactStore, config: OfflineConfig):
        self.meta = meta
        self.store = store
        self.config = config
        self._detector = None
        self._apply_pip_result()

    def _apply_pip_result(self) -> None:
        """Apply the PiP layout detected at ingest time.

        A 50% split gets cropped away (full resolution on the active half);
        a corner inset is added to the ignore list. This happens once per
        analysis run — offline luxury: decide the layout from the entire video,
        not a rolling window.
        """
        if not self.config.pip_autodetect:
            return
        if self.meta.pip_layout is None:
            return
        if self.meta.active_roi is not None and self.config.analysis_roi is None:
            self.config.analysis_roi = self.meta.active_roi
        elif self.meta.pip_region is not None:
            self.config.ignore_regions = [
                *self.config.ignore_regions,
                self.meta.pip_region,
            ]

    def _video_t(self, frame_no: int) -> float:
        """Video time in seconds from frame number (PTS-corrected).

        This is the single timebase swap: every temporal rule in the system
        takes this `t` instead of wall-clock `time.monotonic()`. The analyzers
        are unchanged — they just receive a correct video-time `t`.
        """
        return frame_no / max(self.meta.fps, 0.001)

    def _in_ignore(self, nx: float, ny: float) -> bool:
        for rx, ry, rw, rh in self.config.ignore_regions:
            if rx <= nx <= rx + rw and ry <= ny <= ry + rh:
                return True
        return False

    def _apply_roi(self, frame: np.ndarray) -> np.ndarray:
        roi = self.config.analysis_roi
        if roi is None:
            return frame
        h, w = frame.shape[:2]
        rx, ry, rw, rh = roi
        x0, y0 = int(rx * w), int(ry * h)
        x1, y1 = min(w, int((rx + rw) * w)), min(h, int((ry + rh) * h))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return frame
        return np.ascontiguousarray(frame[y0:y1, x0:x1])

    def _configure_track_buffer(self) -> str:
        """Re-express BoT-SORT's track_buffer from frame count to seconds.

        The live pipeline uses track_buffer: 120 (~8 s at 15 Hz detection).
        For the offline every-frame analysis, this must scale with the video fps:
        track_buffer = max(30, int(fps * track_buffer_s)).
        Writes a per-run temp copy of the tracker YAML rather than patching the
        single git-tracked file in place — native runs would otherwise dirty the
        repo, and two runs on different-fps videos would race on the same file.
        Returns the temp file path; the caller is responsible for removing it.
        """
        import tempfile

        import yaml

        from analysis.detector import TRACKER_YAML

        target = int(max(30, self.meta.fps * self.config.track_buffer_s))
        with open(TRACKER_YAML) as f:
            cfg = yaml.safe_load(f)
        cfg["track_buffer"] = target
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="botsort_drone_")
        with os.fdopen(fd, "w") as f:
            yaml.dump(cfg, f)
        return path

    # ---- P1: Detection pass (stateless, checkpointed/resumable) ----

    def run_pass_p1(self) -> None:
        """P1 detection pass: tiled inference over every frame.

        Persists raw detections (never tracker-adjusted — P1 has no tracker
        at all) plus a raw per-detection appearance embedding (HSV
        histogram) to the sidecar store. The embedding cache is new: the old
        registry.py only kept an EMA-blended gallery, which is insufficient
        for the global re-ID in Phase 3.

        Stateless and checkpointed/resumable: the only thing that crosses
        the checkpoint boundary is the frame/det_id watermark, so a resumed
        run reprocesses from the last persisted frame and reproduces an
        uninterrupted run frame-for-frame, bit-identically.
        """
        pass_name = self.P1_PASS_NAME
        pass_meta = {
            "description": "P1 detection pass — tiled inference over every frame",
            "config": self.config.to_dict(),
            "model_path": self.config.model,
            "device": self.config.device,
            "fps": self.meta.fps,
            "total_frames": self.meta.total_frames,
        }
        self.store.record_pass_start(pass_name, pass_meta)

        # Determinism: seed every RNG up front. Per-frame reseeding below (not
        # persisted RNG state) is what actually makes a resumed run reproduce
        # an uninterrupted run frame-for-frame.
        _seed_rng(self.config.seed)
        try:
            import torch

            torch.use_deterministic_algorithms(True, warn_only=True)
        except ImportError:
            pass

        # Check for resume. Derived purely from the durable frames/detections
        # JSONL on disk, not from the periodic checkpoint file — a crash before
        # the first checkpoint interval fires would otherwise be indistinguishable
        # from a fresh run and reprocess (duplicate) already-persisted frames.
        start_frame = 0
        det_id = 0
        last_frame = self.store.get_last_frame(pass_name)
        if last_frame >= 0:
            # A crash can leave orphaned detection rows for a frame that never
            # got its frame row written (add_frame runs after all of a frame's
            # detections). Discard them before recomputing det_id so that frame
            # is fully redone rather than double-counted.
            self.store.discard_orphaned_detections(pass_name, last_frame)
            start_frame = last_frame + 1
            det_id = self.store.get_max_det_id(pass_name) + 1

        frame_store = FrameStore(self.meta.path, self.meta)
        if start_frame > 0 and not frame_store.seek_to(start_frame):
            msg = f"Could not seek to resume frame {start_frame}"
            self.store.record_pass_error(pass_name, msg)
            frame_store.close()
            return

        # Lazy-load the detector (heavy import)
        from analysis.detector import Detector

        detector = Detector(
            model_path=self.config.model,
            device=self.config.device,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            human_classes=self.config.human_classes,
            threat_classes=self.config.threat_classes,
            tiles=self.config.tiles,
        )

        total_frames = self.meta.total_frames
        processed = 0
        t_start = _time.monotonic()

        try:
            for frame_no in range(start_frame, total_frames):
                frame = frame_store.read()
                if frame is None:
                    break

                # Video time (the timebase swap: wall-clock → video seconds)
                t = self._video_t(frame_no)

                # Apply ROI if configured (for split-screen IR)
                frame = self._apply_roi(frame)
                h, w = frame.shape[:2]

                # Derived per-frame seed: stateless, so a resumed run reseeds
                # identically to an uninterrupted one at this exact frame.
                _seed_rng(_frame_seed(self.config.seed, pass_name, frame_no))

                # Stateless per-frame detection — no tracker involved.
                detections = detector.detect(frame)

                for d in detections:
                    x0, y0, x1, y1 = d.xyxy
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    if self._in_ignore(cx / w, cy / h):
                        continue

                    # Raw per-detection appearance embedding — the cache that
                    # enables global re-ID in Phase 3. Stored per detection, not
                    # EMA-blended, so backward/global association is possible.
                    embedding = None
                    if d.is_human:
                        hist = appearance_hist(frame, d.xyxy)
                        if hist is not None:
                            embedding = hist.tolist()

                    det_record = {
                        "frame_no": frame_no,
                        "det_id": det_id,
                        "xyxy_raw": list(d.xyxy),
                        "conf": round(d.conf, 4),
                        "cls": d.cls_name,
                        "cls_id": d.cls_id,
                        "is_human": d.is_human,
                        "embedding": embedding,
                    }
                    self.store.add_detection(pass_name, frame_no, det_id, det_record)
                    det_id += 1

                # Frame record (minimal for P1 — position data added in P2)
                # PTS computed from the PTS index if available, else frame_no/fps
                frame_record = {
                    "frame_no": frame_no,
                    "pts_ms": round(t * 1000, 3),
                }
                self.store.add_frame(pass_name, frame_no, frame_record)

                processed += 1

                # Checkpoint every ~5 seconds of wall time (cheap enough: one JSON write)
                if processed % max(1, int(self.meta.fps * 5)) == 0:
                    cp = {
                        "pass": pass_name,
                        "last_frame": frame_no,
                        "det_id": det_id,
                        "processed": processed,
                    }
                    self.store.save_checkpoint(pass_name, cp)
                    self.store.record_pass_progress(pass_name, frame_no)
        finally:
            frame_store.close()

        elapsed = _time.monotonic() - t_start
        stats = {
            "frames_processed": processed,
            "total_frames": total_frames,
            "total_detections": det_id,
            "elapsed_s": round(elapsed, 1),
            "fps_effective": round(processed / max(elapsed, 0.001), 1),
        }
        self.store.record_pass_complete(pass_name, stats)

    # ---- P2: Tracking pass (always a full re-run, never checkpointed) ----

    def run_pass_p2(self) -> None:
        """P2 tracking pass: BoT-SORT + GMC driven purely from P1's already-
        persisted detections, plus a fresh decode (GMC needs pixels, not
        re-inference).

        Always re-runs in full — never checkpointed or resumed. It is
        deterministic given the same P1 output, so re-running it twice
        produces byte-identical tracklets, and GMC/track state always
        starts fresh (never carries a stale offset across invocations, the
        way a live scene-cut reset would).

        Persists one row per (track_id, frame) to tracklets/<pass>.jsonl:
        tracklet_id, frame_no, det_id (back-reference to detections/), cls,
        conf, and the tracker/Kalman-adjusted xyxy. detections/xyxy_raw is
        never overwritten — that stays P1's pure predict output.
        """
        pass_name = self.P2_PASS_NAME
        pass_meta = {
            "description": "P2 tracking pass — BoT-SORT + GMC over P1 detections",
            "config": self.config.to_dict(),
            "fps": self.meta.fps,
            "total_frames": self.meta.total_frames,
        }
        self.store.record_pass_start(pass_name, pass_meta)
        self.store.start_fresh_pass_output("tracklets", pass_name)

        dets_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for rec in self.store.iter_detections(self.P1_PASS_NAME):
            dets_by_frame[rec["frame_no"]].append(rec)

        from analysis.tracker import Tracker

        tracker_yaml_path = self._configure_track_buffer()
        total_frames = self.meta.total_frames
        processed = 0
        total_tracklet_rows = 0
        t_start = _time.monotonic()

        frame_store = FrameStore(self.meta.path, self.meta)
        try:
            tracker = Tracker(tracker_yaml_path)
            for frame_no in range(total_frames):
                frame = frame_store.read()
                if frame is None:
                    break
                frame = self._apply_roi(frame)

                # Deterministic order (by det_id, same order P1 persisted
                # them in) — required so BoT-SORT's per-frame output index
                # maps back to the correct det_id.
                records = dets_by_frame.get(frame_no, [])
                for tb in tracker.update(records, frame):
                    self.store.add_tracklet_frame(
                        pass_name,
                        tb.track_id,
                        frame_no,
                        tb.det_id,
                        {
                            "cls": tb.cls_name,
                            "conf": round(tb.conf, 4),
                            "xyxy": list(tb.xyxy),
                        },
                    )
                    total_tracklet_rows += 1
                processed += 1
        finally:
            frame_store.close()
            os.unlink(tracker_yaml_path)

        elapsed = _time.monotonic() - t_start
        stats = {
            "frames_processed": processed,
            "total_frames": total_frames,
            "total_tracklet_rows": total_tracklet_rows,
            "elapsed_s": round(elapsed, 1),
            "fps_effective": round(processed / max(elapsed, 0.001), 1),
        }
        self.store.record_pass_complete(pass_name, stats)
