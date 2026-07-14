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
from typing import TYPE_CHECKING, Any

import numpy as np

from analysis.ingest import FrameStore, VideoMeta
from analysis.store import ArtifactStore

if TYPE_CHECKING:
    # Avoid a runtime circular import: identity imports OfflineConfig from
    # this module, so AssociationResult is only needed for the P3 return type.
    from analysis.identity import AssociationResult

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
    # Inference-time cutoff for what P1 persists. Deliberately matches BoT-SORT's
    # track_low_thresh (analysis/trackers/botsort_drone.yaml) rather than a
    # display-quality threshold, so P2's low-score second-association (BYTE)
    # bucket — [track_low_thresh, track_high_thresh) — actually receives data.
    # Filtering for display/analysis is a separate, higher threshold: display_conf.
    detect_conf: float = 0.08
    display_conf: float = 0.30
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

    # ---- P3 identity (global tracklet association) ----
    # Per-detection appearance embedding. None/missing weights → HSV-only
    # embedder (the carried-forward appearance_hist). A weights path present
    # on disk → OSNet primary with HSV fallback below reid_floor. The floor
    # is a deliberate, documented degradation for small/distant people at
    # altitude (10 px crops are below any ReID model's input), not a bug.
    reid_weights: str | None = None
    reid_floor: int = 32  # min crop side (px) below which the HSV fallback fires

    # P3 clustering gates. The spatio-temporal gate generalizes the live
    # registry's reid_max_dist_frac × diag × (1+gap) rule (registry.py
    # _match_lost); the temporal-overlap exclusion is the offline-only hard
    # constraint that is impossible live (live never has the full frame set).
    p3_sim_thresh: float = 0.86  # appearance cosine to merge two tracklets
    p3_min_gap_s: float = 0.0  # offline, non-overlap is the real constraint
    # (the live 0.3s proximity heuristic is moot once
    # the full frame set is available — overlap exclusion
    # subsumes it); kept as a knob for jitter suppression
    p3_max_gap_s: float = 60.0  # beyond this, a re-entry is implausible
    p3_max_dist_frac: float = 0.45  # dist ≤ frac × diag × (1 + gap_s)
    p3_confirm_s: float = 2.0  # a person counts only after existing this long
    # Margin below p3_sim_thresh that still counts as an "uncertain" near-
    # merge for the honesty band ("N unika, varav M osäkra sammanslagningar").
    p3_uncertain_margin: float = 0.04

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
            "detect_conf": self.detect_conf,
            "display_conf": self.display_conf,
            "iou": self.iou,
            "tiles": self.tiles,
            "human_classes": sorted(self.human_classes),
            "pip_autodetect": self.pip_autodetect,
            "seed": self.seed,
            "track_buffer_s": self.track_buffer_s,
            "reid_weights": self.reid_weights,
            "reid_floor": self.reid_floor,
            "p3_sim_thresh": self.p3_sim_thresh,
            "p3_min_gap_s": self.p3_min_gap_s,
            "p3_max_gap_s": self.p3_max_gap_s,
            "p3_max_dist_frac": self.p3_max_dist_frac,
            "p3_confirm_s": self.p3_confirm_s,
            "p3_uncertain_margin": self.p3_uncertain_margin,
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
        at all) plus a per-detection appearance embedding to the sidecar
        store. The embedding is the identity substrate P3 clusters over:
        OSNet when weights are present, HSV otherwise, with HSV as the
        deliberate fallback for crops below the ReID floor (see
        analysis.embedding). The `embedding_method` field tags each
        detection so a consumer knows which space its vector is in.

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
        from analysis.embedding import make_embedder

        detector = Detector(
            model_path=self.config.model,
            device=self.config.device,
            imgsz=self.config.imgsz,
            conf=self.config.detect_conf,
            iou=self.config.iou,
            human_classes=self.config.human_classes,
            threat_classes=self.config.threat_classes,
            tiles=self.config.tiles,
        )
        # Embedder is constructed once (the OSNet model load is expensive);
        # imported lazily so tests can swap in a FakeEmbedder the same way
        # they swap FakeDetector in for the Detector.
        embedder = make_embedder(self.config)

        total_frames = self.meta.total_frames
        processed = 0
        ended_early_at: int | None = None
        t_start = _time.monotonic()

        try:
            for frame_no in range(start_frame, total_frames):
                frame = frame_store.read()
                if frame is None:
                    ended_early_at = frame_no
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

                    # Per-detection appearance embedding — the identity
                    # substrate P3 clusters over. Stored per detection, not
                    # EMA-blended, so backward/global association is possible
                    # (the live registry only kept a blended gallery). The
                    # method tag records which embedding space this vector is
                    # in: "osnet" (dedicated ReID) or "hsv" (the fallback for
                    # crops below the ReID floor). Non-human detections and
                    # degenerate crops get a null embedding + method.
                    embedding = None
                    embedding_method = None
                    if d.is_human:
                        result = embedder.embed(frame, d.xyxy)
                        if result is not None:
                            embedding = result.vector.tolist()
                            embedding_method = result.method

                    det_record = {
                        "frame_no": frame_no,
                        "det_id": det_id,
                        "xyxy_raw": list(d.xyxy),
                        "conf": round(d.conf, 4),
                        "cls": d.cls_name,
                        "cls_id": d.cls_id,
                        "is_human": d.is_human,
                        "embedding": embedding,
                        "embedding_method": embedding_method,
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
        if ended_early_at is not None:
            msg = f"decode ended early at frame {ended_early_at} of {total_frames}"
            self.store.record_pass_error(pass_name, msg)
        else:
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

    # ---- P3: Identity pass (global tracklet association) ----

    P3_PASS_NAME = "p3_identity"

    def _effective_dims(self) -> tuple[int, int]:
        """The frame size P1/P2 actually processed (after optional ROI crop),
        without decoding a frame. P3 uses this for the frame diagonal that
        scales the spatio-temporal gate."""
        w, h = self.meta.width, self.meta.height
        roi = self.config.analysis_roi
        if roi is not None:
            rx, ry, rw, rh = roi
            x0, y0 = int(rx * w), int(ry * h)
            x1, y1 = min(w, int((rx + rw) * w)), min(h, int((ry + rh) * h))
            if x1 - x0 >= 32 and y1 - y0 >= 32:
                w, h = x1 - x0, y1 - y0
        return w, h

    def run_pass_p3(self) -> "AssociationResult | None":
        """P3 identity pass: global re-association of P2 tracklets into persons.

        Reads P1's per-detection embeddings (joined to P2's tracklets via
        det_id), aggregates each tracklet's appearance into a per-method
        centroid, then runs constrained agglomerative clustering — merging
        tracklet pairs into persons gated by hard temporal-overlap exclusion,
        spatio-temporal plausibility, and appearance similarity (see
        analysis.identity for the full design).

        Like P2, this pass is never checkpointed and always re-runs in full:
        it is deterministic given the same P1+P2 output and consumes no
        inference, so re-running produces byte-identical persons + assoc_audit.
        Returns the AssociationResult (or None if P2 isn't complete and the
        gate refused to run).
        """
        pass_name = self.P3_PASS_NAME
        pass_meta = {
            "description": "P3 identity — global tracklet association into persons",
            "config": self.config.to_dict(),
            "fps": self.meta.fps,
        }
        self.store.record_pass_start(pass_name, pass_meta)

        # Gate on P2 success, the same way the CLI gates P2 on P1: refusing to
        # run P3 over an incomplete artifact prevents a persons/ table built
        # from a truncated tracklet set that the manifest would then present
        # as final.
        p2_info = self.store._manifest.get("passes", {}).get(self.P2_PASS_NAME, {})
        if p2_info.get("status") != "complete":
            self.store.record_pass_error(pass_name, f"P2 not complete (status: {p2_info.get('status')})")
            return None

        from analysis.identity import associate, build_tracklet_profiles

        self.store.start_fresh_pass_output("persons", pass_name)

        tracklet_rows = list(self.store.iter_tracklets(self.P2_PASS_NAME))
        detection_by_id = {rec["det_id"]: rec for rec in self.store.iter_detections(self.P1_PASS_NAME)}

        t_start = _time.monotonic()
        profiles = build_tracklet_profiles(tracklet_rows, detection_by_id, self.meta.fps)
        w, h = self._effective_dims()
        frame_diag = float(np.hypot(w, h))
        result = associate(profiles, self.config, frame_diag)

        for person in result.persons:
            self.store.add_person(
                pass_name,
                person.person_id,
                {
                    "tracklet_ids": person.tracklet_ids,
                    "embedding_centroids": person.embedding_centroids,
                    "embedding_counts": person.embedding_counts,
                    "first_seen": person.first_seen,
                    "last_seen": person.last_seen,
                    "confirmation_state": person.confirmation_state,
                    "assoc_audit": person.assoc_audit,
                },
            )

        elapsed = _time.monotonic() - t_start
        stats = {
            "tracklets_in": len(profiles),
            "persons_out": len(result.persons),
            "confirmed_persons": result.confirmed_count,
            "uncertain_merges": result.uncertain_merges,
            "frame_diag": round(frame_diag, 1),
            "elapsed_s": round(elapsed, 3),
        }
        self.store.record_pass_complete(pass_name, stats)
        return result

    # ---- P5: Event derivation pass (always a full re-run) ----

    P5_PASS_NAME = "p5_events"

    def run_pass_p5(self) -> int:
        """P5 event derivation: replay BehaviorAnalyzer + SituationAnalyzer
        over P2's tracklets (+ the frame stream) and diff their per-frame
        status into discrete STILLA / MOT_FARA / HAZARD events.

        This is the marriage of the report's P4 (per-frame behavior/situation
        status) and P5 (status diffing): the analyzers are stateless per-
        call, so there's no value in persisting their per-frame output
        separately — we compute and diff in one pass.

        Person-keyed events (STILLA/MOT_FARA) are tagged with P3's person_id
        when P3 ran; otherwise person_id is null. HAZARD events are never
        person-keyed.

        Like P2/P3, always re-runs in full and is deterministic given the
        same P1+P2(+P3) output. No inference is re-run; only the cheap
        behavior/situation heuristics are replayed over the persisted
        tracklets and a fresh frame decode (the situation analyzer needs
        pixels). Returns the number of events emitted.

        IRRATIONELL is explicitly Phase 4 per the report's build order and is
        not derived here — the sub-signal set in §4 will slot in as another
        status stream feeding the same diff, without restructuring this pass.
        """
        pass_name = self.P5_PASS_NAME
        pass_meta = {
            "description": "P5 event derivation — behavior/situation status diffed into events",
            "config": self.config.to_dict(),
            "fps": self.meta.fps,
        }
        self.store.record_pass_start(pass_name, pass_meta)
        self.store.start_fresh_pass_output("events", pass_name)

        # Gate on P2: behavior events need tracklets. P3 is optional; when it
        # didn't run (--no-p3) person_id stays null on every event.
        p2_info = self.store._manifest.get("passes", {}).get(self.P2_PASS_NAME, {})
        if p2_info.get("status") != "complete":
            self.store.record_pass_error(pass_name, f"P2 not complete (status: {p2_info.get('status')})")
            return 0

        # Build the tracklet→person map from P3 if present.
        person_by_tracklet: dict[int, int] = {}
        p3_info = self.store._manifest.get("passes", {}).get(self.P3_PASS_NAME, {})
        if p3_info.get("status") == "complete":
            for person in self.store.iter_persons(self.P3_PASS_NAME):
                for tid in person.get("tracklet_ids", []):
                    person_by_tracklet[int(tid)] = int(person["person_id"])

        from analysis.events import derive_events

        tracklet_rows = list(self.store.iter_tracklets(self.P2_PASS_NAME))
        w, h = self._effective_dims()

        frame_store = FrameStore(self.meta.path, self.meta)
        t_start = _time.monotonic()
        total_events = 0
        try:
            # The situation analyzer needs the frame stream; we feed frames
            # by re-decoding (same as P2). ROI is applied to match P1/P2.
            def _frames():
                for _ in range(self.meta.total_frames):
                    frame = frame_store.read()
                    if frame is None:
                        break
                    yield self._apply_roi(frame)

            events = derive_events(
                tracklet_rows,
                person_by_tracklet=person_by_tracklet,
                frames=_frames(),
                fps=self.meta.fps,
                frame_w=w,
                frame_h=h,
                config=self.config,
                ignore_regions=list(self.config.ignore_regions),
            )
        finally:
            frame_store.close()

        for ev in events:
            self.store.add_event(pass_name, ev.event_id, ev.to_dict())
            total_events += 1

        # Per-category breakdown for the manifest (handy for the review UI's
        # summary header without having to walk the events file).
        by_cat: dict[str, int] = {}
        for ev in events:
            by_cat[ev.category] = by_cat.get(ev.category, 0) + 1
        elapsed = _time.monotonic() - t_start
        stats = {
            "events_out": total_events,
            "by_category": by_cat,
            "p3_used": bool(person_by_tracklet),
            "elapsed_s": round(elapsed, 3),
        }
        self.store.record_pass_complete(pass_name, stats)
        return total_events
