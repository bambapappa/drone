"""Analysis pipeline: two threads per source (see DECISIONS.md B3).

Render thread (~max_fps): reads frames, estimates camera motion, carries
boxes along with local optical flow, smooths display boxes with a
flow-fed-forward EMA + slew filter, encodes JPEG and broadcasts
[meta-JSON][JPEG] packets.

Detect thread (as fast as the CPU allows): YOLO + BoT-SORT on the latest
frame, person registry (re-ID), behavior analysis, situation assessment,
threat flagging. Results are merged into the render thread's display tracks,
compensated for the motion that happened while detection was running.
"""

from __future__ import annotations

import json
import math
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.config import Settings
from app.vision.behavior import BehaviorAnalyzer, BehaviorConfig
from app.vision.broadcast import Broadcaster
from app.vision.flow import BoxFilter, GlobalMotion, local_box_flow
from app.vision.pip import PipAutoDetector, split_active_roi
from app.vision.registry import PersonRegistry, appearance_hist
from app.vision.situation import SituationAnalyzer
from app.vision.sources import VideoSource

FLOW_W = 480  # gray working width for box flow


@dataclass
class DisplayTrack:
    track_id: int
    box: tuple[float, float, float, float]  # raw: flow-advected + detection-corrected (source px)
    filt: BoxFilter
    disp: tuple[float, float, float, float] = (0, 0, 0, 0)  # displayed: smoothed raw, every frame
    pid: int | None = None
    cls_name: str = "person"
    is_threat: bool = False
    conf: float = 0.0
    status: str = "ok"
    prone: bool = False
    speed: float = 0.0
    flow_acc: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_det_t: float = 0.0
    trail: deque = field(default_factory=lambda: deque(maxlen=32))  # stab points


@dataclass
class DetJob:
    frame: np.ndarray
    t: float
    frame_no: int
    global_offset: np.ndarray
    track_flow_acc: dict[int, np.ndarray]


class Pipeline:
    def __init__(self, cfg: Settings, source: str, broadcaster: Broadcaster):
        self.cfg = cfg
        self.broadcaster = broadcaster
        self.source = VideoSource(source, loop=cfg.loop, max_fps=cfg.max_fps)
        self.source_name = source
        self.roi = cfg.roi_tuple()  # crop applied to every frame before analysis
        self.ignore = cfg.ignore_list()  # regions excluded from analysis (PiP IR insets)
        # Auto-detect an IR inset only when the operator hasn't pinned regions.
        self._pip = PipAutoDetector() if (cfg.pip_autodetect and not self.ignore) else None
        self.pip_layout: str | None = None
        self._pip_applied = self._pip is None
        self._pip_frame_ctr = 0
        self.frame_w = 0  # dimensions of the (possibly cropped) analyzed frame
        self.frame_h = 0

        self.gm = GlobalMotion()
        self.registry = PersonRegistry(
            sim_thresh=cfg.reid_sim_thresh,
            max_gap_s=cfg.reid_max_gap_s,
            max_dist_frac=cfg.reid_max_dist_frac,
        )
        self.behavior = BehaviorAnalyzer(
            BehaviorConfig(
                window_s=cfg.beh_window_s,
                min_history_s=cfg.beh_min_history_s,
                still_speed=cfg.beh_still_speed,
                still_time_s=cfg.beh_still_time_s,
                toward_speed=cfg.beh_toward_speed,
                toward_angle_deg=cfg.beh_toward_angle_deg,
                toward_time_s=cfg.beh_toward_time_s,
                prone_aspect=cfg.beh_prone_aspect,
            )
        )
        self.situation = SituationAnalyzer(
            min_area=cfg.hazard_min_area,
            hold_s=cfg.hazard_hold_s,
            flow_ema=cfg.smoke_flow_ema,
            base_margin=cfg.base_margin,
            base_hysteresis=cfg.base_hysteresis,
            fire_require_smoke=cfg.fire_require_smoke,
        )

        self._tracks: dict[int, DisplayTrack] = {}
        self._danger_stab: tuple[float, float] | None = None
        self._irrational_pids: set[int] = set()  # cumulative over session

        self._job: DetJob | None = None
        self._job_cv = threading.Condition()
        self._result_lock = threading.Lock()
        self._result: dict | None = None

        self._stop = threading.Event()
        self._tracker_reset = threading.Event()
        self._render_thread: threading.Thread | None = None
        self._detect_thread: threading.Thread | None = None

        self.status = "starting"
        self.error: str | None = None
        self.render_fps = 0.0
        self.detect_fps = 0.0
        self._threat_last_t = 0.0
        self._last_threats: list[dict] = []

    # ---------- public control ----------

    def start(self) -> None:
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True, name="render")
        self._detect_thread = threading.Thread(target=self._detect_loop, daemon=True, name="detect")
        self._render_thread.start()
        self._detect_thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._job_cv:
            self._job_cv.notify_all()
        for th in (self._render_thread, self._detect_thread):
            if th is not None:
                th.join(timeout=5.0)
        self.source.release()
        self.status = "stopped"

    def set_danger_norm(self, pos: tuple[float, float] | None) -> None:
        """Danger point in normalized screen coords -> stored camera-stabilized."""
        if pos is None:
            self._danger_stab = None
            return
        x, y = pos[0] * max(self.frame_w, 1), pos[1] * max(self.frame_h, 1)
        self._danger_stab = self.gm.to_stab(x, y)

    def danger_screen_norm(self) -> tuple[float, float, bool] | None:
        if self._danger_stab is None or self.frame_w == 0:
            return None
        x, y = self.gm.to_screen(*self._danger_stab)
        nx, ny = x / self.frame_w, y / self.frame_h
        off = not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0)
        return min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0), off

    # ---------- render thread ----------

    def _render_loop(self) -> None:
        """Render thread: the display heartbeat. Each frame it estimates camera
        motion, advects boxes with optical flow, merges any fresh detection,
        and broadcasts a JPEG+metadata packet — all at display rate, decoupled
        from the slower detect thread so boxes never stall."""
        try:
            if not self.source.open():
                self.status = "error"
                self.error = f"Kunde inte öppna källan: {self.source_name}"
                return
            self.status = "running"
            prev_small_gray: np.ndarray | None = None
            scale = 1.0
            fps_ema = 0.0
            last_t = time.monotonic()

            while not self._stop.is_set():
                frame = self.source.read()
                if frame is None:
                    self.status = "ended"
                    break
                t = time.monotonic()
                if self.source.just_looped:
                    self.source.just_looped = False
                    self._on_scene_cut()
                    prev_small_gray = None
                frame = self._apply_roi(frame)
                h, w = frame.shape[:2]
                self.frame_w, self.frame_h = w, h
                # Sample for an IR inset a few times a second (it may appear
                # partway into a feed); apply once a layout locks.
                if self._pip is not None and not self._pip_applied:
                    self._pip_frame_ctr += 1
                    if self._pip_frame_ctr % 8 == 0 and self._pip.feed(frame):
                        self._apply_pip_result()
                scale = w / float(FLOW_W)
                small_gray = cv2.cvtColor(
                    cv2.resize(frame, (FLOW_W, max(2, int(h / scale)))), cv2.COLOR_BGR2GRAY
                )
                self.gm.update(small_gray, scale)
                gshift = self.gm.last_shift

                if prev_small_gray is not None and prev_small_gray.shape == small_gray.shape:
                    self._advect_tracks(prev_small_gray, small_gray, scale, gshift, t)
                prev_small_gray = small_gray

                self._merge_detections(t)
                self._submit_job(frame, t)
                self._prune_tracks(t)

                dt = t - last_t
                last_t = t
                if dt > 0:
                    fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt
                self.render_fps = fps_ema

                # Analysis always runs (counts, behavior, situation keep
                # accumulating) but encoding is wasted work with no viewers.
                if self.broadcaster.client_count > 0:
                    packet = self._build_packet(frame, t)
                    self.broadcaster.publish(packet)
        except Exception as e:  # surface, don't die silently
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"
        finally:
            with self._job_cv:
                self._stop.set()
                self._job_cv.notify_all()

    def _advect_tracks(
        self, prev_gray, cur_gray, scale: float, gshift: tuple[float, float], t: float
    ) -> None:
        fallback = (gshift[0] / scale, gshift[1] / scale)
        for tr in self._tracks.values():
            cx, cy, bw, bh = tr.box
            small_box = (cx / scale, cy / scale, bw / scale, bh / scale)
            dx, dy = local_box_flow(prev_gray, cur_gray, small_box, fallback)
            dx, dy = dx * scale, dy * scale
            tr.box = (cx + dx, cy + dy, bw, bh)
            tr.disp = tr.filt(tr.box, t, ff=(dx, dy))
            tr.flow_acc += (dx, dy)

    def _submit_job(self, frame: np.ndarray, t: float) -> None:
        job = DetJob(
            frame=frame,
            t=t,
            frame_no=self.source.frame_no,
            global_offset=self.gm.offset.copy(),
            track_flow_acc={tid: tr.flow_acc.copy() for tid, tr in self._tracks.items()},
        )
        with self._job_cv:
            self._job = job  # newest wins; detect thread always takes latest
            self._job_cv.notify()

    def _merge_detections(self, t: float) -> None:
        """Fold the detect thread's latest result into the display tracks.

        Detection ran on a frame that is now several frames old, during which
        the camera panned and people moved. We therefore shift each detection
        forward to "now" before correcting the display box to it:
          - existing track: by how far that track's optical flow carried it
            since the job was submitted (`flow_acc - acc0`);
          - new track: by the global camera drift since (`gm.offset - offset0`).
        Without this the box would snap backward to where the person was when
        detection started. The corrected target is then eased in by the box
        filter (slew-limited) so it glides rather than teleports.
        """
        with self._result_lock:
            res = self._result
            self._result = None
        if res is None:
            return
        seen_ids = {d["track_id"] for d in res["detections"] if d["track_id"] is not None}
        for d in res["detections"]:
            tid = d["track_id"]
            if tid is None:
                continue
            x0, y0, x1, y1 = d["xyxy"]
            target = ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
            tr = self._tracks.get(tid)
            is_new = tr is None
            if is_new:
                # Appeared while detection ran: place at detection pos + global drift since.
                gd = self.gm.offset - res["global_offset"]
                target = (target[0] + gd[0], target[1] + gd[1], target[2], target[3])
                tr = DisplayTrack(
                    track_id=tid,
                    box=target,
                    filt=BoxFilter(self.cfg.smooth_tau_pos, self.cfg.smooth_tau_size, self.cfg.smooth_slew),
                )
                tr.disp = tr.filt(target, t)
                self._tracks[tid] = tr
            else:
                # Compensate for motion this track made while detection ran.
                acc0 = res["track_flow_acc"].get(tid)
                if acc0 is not None:
                    fd = tr.flow_acc - acc0
                else:
                    fd = self.gm.offset - res["global_offset"]
                target = (target[0] + fd[0], target[1] + fd[1], target[2], target[3])
                if t - tr.last_det_t > self._track_grace():
                    # Re-acquired after being undetected for a while (e.g. the
                    # person left the frame): snap state, don't drag the filter
                    # across the gap. The track was pruned from display anyway.
                    tr.filt.reset_to(target, t)
                    tr.disp = target
                tr.box = target
            # A re-identified person gets a new tracker id; kill any stale
            # advected track still rendering under the same pid, and glide
            # the new box from the old position instead of teleporting.
            if d["pid"] is not None:
                for otid, otr in list(self._tracks.items()):
                    if otid != tid and otr.pid == d["pid"] and otid not in seen_ids:
                        tr.trail = deque(list(otr.trail) + list(tr.trail), maxlen=tr.trail.maxlen)
                        if is_new:
                            tr.filt.reset_to(otr.disp, t - 1.0 / max(self.render_fps, 10.0))
                            tr.disp = tr.filt(target, t)
                        del self._tracks[otid]
            tr.pid = d["pid"]
            tr.cls_name = d["cls_name"]
            tr.is_threat = d["is_threat"]
            tr.conf = d["conf"]
            tr.status = d["status"]
            tr.prone = d["prone"]
            tr.speed = d["speed"]
            tr.last_det_t = t
            if d["trail_pt"] is not None:
                tr.trail.append(d["trail_pt"])

        if res["threats"]:
            # Keep the last seen threats during the hold window so the alarm
            # doesn't flicker with single missed detections.
            self._last_threats = res["threats"]
            self._threat_last_t = t
        self._irrational_pids |= res["irrational_pids"]

    def _apply_pip_result(self) -> None:
        """Adopt the auto-detected IR inset: a 50% split is cropped away (full
        resolution kept on the active half), a corner inset is masked out."""
        self._pip_applied = True
        if self._pip is None or self._pip.region is None:
            return
        self.pip_layout = self._pip.layout
        active = split_active_roi(self._pip.layout or "")
        if active is not None and self.roi is None:
            # Crop to the real half. Dims change, so reset motion/danger/tracks.
            self.roi = active
            self._danger_stab = None
            self.gm = GlobalMotion()
            self._on_scene_cut()
        else:
            # Corner inset (or split when a manual ROI is already set): mask it.
            self.ignore = [*self.ignore, self._pip.region]

    def _in_ignore(self, nx: float, ny: float) -> bool:
        for rx, ry, rw, rh in self.ignore:
            if rx <= nx <= rx + rw and ry <= ny <= ry + rh:
                return True
        return False

    def _apply_roi(self, frame: np.ndarray) -> np.ndarray:
        """Crop to the configured analysis region (e.g. the visual half of
        split-screen IR footage) so all downstream stages see one consistent
        view and people aren't double-counted."""
        if self.roi is None:
            return frame
        h, w = frame.shape[:2]
        rx, ry, rw, rh = self.roi
        x0, y0 = int(rx * w), int(ry * h)
        x1, y1 = min(w, int((rx + rw) * w)), min(h, int((ry + rh) * h))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return frame
        return np.ascontiguousarray(frame[y0:y1, x0:x1])

    def _on_scene_cut(self) -> None:
        """File loop restart (or equivalent discontinuity): drop all visual
        tracking state. Person identities live in the registry and survive —
        people are re-identified by appearance when redetected."""
        self._tracks.clear()
        self.gm.reset_motion()
        self.behavior.drop_inactive(set())
        self._tracker_reset.set()
        with self._result_lock:
            self._result = None

    def _track_grace(self) -> float:
        """How long a track may stay on screen without a fresh detection:
        a few detection periods (bridges single misses), capped so ghost
        boxes never linger. The cap widens when detection is slow (large
        models on weak CPUs) — flow advection carries the boxes meanwhile."""
        return min(max(3.5 / max(self.detect_fps, 1.0), 0.35), 2.5)

    def _prune_tracks(self, t: float) -> None:
        grace = self._track_grace()
        for tid in list(self._tracks):
            if t - self._tracks[tid].last_det_t > grace:
                del self._tracks[tid]

    # ---------- detect thread ----------

    def _detect_loop(self) -> None:
        """Detect thread: load the model once, then repeatedly take the newest
        submitted frame, run YOLO+tracking+analysis, and publish the result for
        the render thread to merge. Model import/load is lazy so the web app and
        unit tests don't require torch."""
        try:
            from app.vision.detector import Detector

            detector = Detector(
                model_path=self.cfg.model,
                device=self.cfg.device,
                imgsz=self.cfg.imgsz,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                human_classes=self.cfg.human_class_set(),
                threat_classes=self.cfg.threat_class_set(),
                tiles=self.cfg.tiles,
            )
        except Exception as e:
            self.status = "error"
            self.error = f"Modellfel: {type(e).__name__}: {e}"
            self._stop.set()
            return

        fps_ema = 0.0
        last_done = time.monotonic()
        while not self._stop.is_set():
            with self._job_cv:
                while self._job is None and not self._stop.is_set():
                    self._job_cv.wait(timeout=0.5)
                job, self._job = self._job, None
            if job is None or self._stop.is_set():
                continue
            try:
                if self._tracker_reset.is_set():
                    self._tracker_reset.clear()
                    detector.reset_tracker()
                detections = detector.track(job.frame)
                result = self._analyze(job, detections)
            except Exception as e:
                self.error = f"Analysfel: {type(e).__name__}: {e}"
                continue
            with self._result_lock:
                self._result = result
            now = time.monotonic()
            dt = now - last_done
            last_done = now
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt
            self.detect_fps = fps_ema

    def _analyze(self, job: DetJob, detections) -> dict:
        h, w = job.frame.shape[:2]
        diag = math.hypot(w, h)
        offset = job.global_offset
        self.registry.begin_frame()

        danger_stab = self._danger_stab
        det_out: list[dict] = []
        threats: list[dict] = []
        irrational: set[int] = set()
        visible = 0
        active_pids: set[int] = set()

        for d in detections:
            x0, y0, x1, y1 = d.xyxy
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if self._in_ignore(cx / w, cy / h):
                continue
            stab = (cx - float(offset[0]), cy - float(offset[1]))
            entry = {
                "track_id": d.track_id,
                "xyxy": d.xyxy,
                "cls_name": d.cls_name,
                "conf": d.conf,
                "is_threat": d.is_threat and not d.is_human,
                "pid": None,
                "status": "ok",
                "prone": False,
                "speed": 0.0,
                "trail_pt": None,
            }
            if d.is_human and d.track_id is not None:
                visible += 1
                hist = appearance_hist(job.frame, d.xyxy)
                pid = self.registry.resolve(d.track_id, job.t, hist, stab, diag)
                active_pids.add(pid)
                bw, bh = x1 - x0, max(y1 - y0, 1.0)
                status, prone, speed = self.behavior.update(pid, job.t, stab, bh, bw / bh, danger_stab)
                entry.update(pid=pid, status=status, prone=prone, speed=speed, trail_pt=stab)
                if status != "ok":
                    irrational.add(pid)
            if entry["is_threat"]:
                threats.append(
                    {
                        "box": self._norm_box(d.xyxy, w, h),
                        "cls": d.cls_name,
                        "conf": round(d.conf, 2),
                    }
                )
            det_out.append(entry)

        danger_norm = None
        ds = self.danger_screen_norm()
        if ds is not None:
            danger_norm = (ds[0], ds[1])
        self.situation.update(job.frame, job.t, danger_norm, ignore=self.ignore)

        return {
            "detections": det_out,
            "threats": threats,
            "visible": visible,
            "irrational_pids": irrational,
            "global_offset": offset,
            "track_flow_acc": job.track_flow_acc,
        }

    # ---------- packet ----------

    @staticmethod
    def _norm_box(xyxy, w: int, h: int) -> list[float]:
        x0, y0, x1, y1 = xyxy
        return [
            round(x0 / w, 4),
            round(y0 / h, 4),
            round((x1 - x0) / w, 4),
            round((y1 - y0) / h, 4),
        ]

    def _build_packet(self, frame: np.ndarray, t: float) -> bytes:
        h, w = frame.shape[:2]
        persons = []
        threat_boxes = []
        irr_now = 0
        for tr in self._tracks.values():
            cx, cy, bw, bh = tr.disp
            nb = [
                round((cx - bw / 2) / w, 4),
                round((cy - bh / 2) / h, 4),
                round(bw / w, 4),
                round(bh / h, 4),
            ]
            if tr.is_threat:
                threat_boxes.append({"box": nb, "cls": tr.cls_name, "conf": round(tr.conf, 2)})
                continue
            if tr.pid is None:
                continue
            if tr.status != "ok":
                irr_now += 1
            trail = [
                [round((sx + self.gm.offset[0]) / w, 4), round((sy + self.gm.offset[1]) / h, 4)]
                for sx, sy in list(tr.trail)[-24:]
            ]
            persons.append(
                {
                    "pid": tr.pid,
                    "tid": tr.track_id,
                    "box": nb,
                    "conf": round(tr.conf, 2),
                    "st": tr.status,
                    "prone": tr.prone,
                    "sp": round(tr.speed, 2),
                    "trail": trail,
                }
            )

        st = self.situation.state
        threat_active = (t - self._threat_last_t) < self.cfg.threat_hold_s and bool(self._last_threats)
        danger = self.danger_screen_norm()
        meta = {
            "v": 1,
            "t": round(t, 3),
            "wh": [w, h],
            "fps": round(self.render_fps, 1),
            "det_fps": round(self.detect_fps, 1),
            "persons": persons,
            "threats": threat_boxes,
            "hazards": {
                "fire": {"pos": list(st.fire.pos), "area": round(st.fire.area, 4)} if st.fire else None,
                "smoke": (
                    {
                        "pos": list(st.smoke.pos),
                        "area": round(st.smoke.area, 4),
                        "drift": [round(v, 5) for v in st.smoke_drift],
                    }
                    if st.smoke
                    else None
                ),
            },
            "base": {"pos": list(st.base), "reasons": st.base_reasons} if st.base else None,
            "danger": {"pos": [danger[0], danger[1]], "off": danger[2]} if danger else None,
            "stats": {
                "unique": self.registry.unique_total,
                # What the viewer actually sees right now (display tracks),
                # not the last raw detection count.
                "visible": len(persons),
                "irr_now": irr_now,
                "irr_total": len(self._irrational_pids),
                "threat": threat_active,
            },
            "src": {
                "name": self.source_name,
                "live": self.source.is_live,
                "status": self.status,
            },
        }

        out_w = min(self.cfg.out_width, w)
        if out_w < w:
            small = cv2.resize(frame, (out_w, int(h * out_w / w)))
        else:
            small = frame
        ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        if not ok:
            jpeg = np.zeros(0, dtype=np.uint8)
        meta_b = json.dumps(meta, separators=(",", ":")).encode()
        return struct.pack(">I", len(meta_b)) + meta_b + jpeg.tobytes()


class PipelineManager:
    """Owns the single active pipeline; switches sources safely."""

    def __init__(self, cfg: Settings, broadcaster: Broadcaster):
        self.cfg = cfg
        self.broadcaster = broadcaster
        self.pipeline: Pipeline | None = None
        self._lock = threading.Lock()

    def start(self, source: str) -> None:
        with self._lock:
            if self.pipeline is not None:
                self.pipeline.stop()
            self.pipeline = Pipeline(self.cfg, source, self.broadcaster)
            self.pipeline.start()

    def stop(self) -> None:
        with self._lock:
            if self.pipeline is not None:
                self.pipeline.stop()
                self.pipeline = None

    def state(self) -> dict:
        p = self.pipeline
        if p is None:
            return {"status": "idle", "source": None, "error": None}
        return {
            "status": p.status,
            "source": p.source_name,
            "error": p.error,
            "render_fps": round(p.render_fps, 1),
            "detect_fps": round(p.detect_fps, 1),
            "unique_total": p.registry.unique_total,
            "clients": self.broadcaster.client_count,
            "pip_layout": p.pip_layout,  # auto-detected IR inset, or None
        }
