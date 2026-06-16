"""Offline (non-real-time) analysis: run the full model on every frame.

The live pipeline (pipeline.py) trades detection quality for latency: it
detects on a background thread at a few Hz and carries boxes between
detections with optical flow. That is the right call for a live drone feed,
but for *after-action review* we don't care about latency — we care about
the best possible annotation. So this analyzer runs detection on **every
frame** (no flow advection, no realtime pacing) and writes a self-contained
"bundle" that a scrubbable web player replays:

    <out_dir>/meta.json    — header (source, fps, model, config, summary)
    <out_dir>/frames.jsonl — one metadata record per analyzed frame
    <out_dir>/events.json  — timeline of notable transitions (post-analysis)
    <out_dir>/state.json    — progress, written live while running

The per-frame record uses the **same schema** the live WebSocket sends, so
the player reuses the exact overlay drawing code (static/overlay.js). The
imagery is not duplicated into the bundle — the player streams the original
video file and overlays annotations synced by time.

Reuses the production vision components (detector, registry, behavior,
situation, global motion) so offline and live agree on what they see.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings
from app.vision.behavior import STATUS_OK, BehaviorAnalyzer, BehaviorConfig
from app.vision.flow import GlobalMotion
from app.vision.registry import PersonRegistry, appearance_hist
from app.vision.situation import SituationAnalyzer

FLOW_W = 480  # gray working width for global-motion estimate (matches pipeline)
TRAIL_MAX = 32


def _norm_xywh(xyxy, w: int, h: int) -> list[float]:
    x0, y0, x1, y1 = xyxy
    return [round(x0 / w, 4), round(y0 / h, 4), round((x1 - x0) / w, 4), round((y1 - y0) / h, 4)]


class OfflineAnalyzer:
    """Frame-by-frame analyzer producing a replayable bundle.

    The detector is created lazily (so importing this module doesn't pull in
    torch) and can be injected for tests via the ``detector`` argument.
    """

    def __init__(self, cfg: Settings, source_path: str, *, detector=None):
        self.cfg = cfg
        self.source_path = source_path
        self.roi = cfg.roi_tuple()
        self.ignore = cfg.ignore_list()
        self._detector = detector

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
        self._trails: dict[int, deque] = {}
        self._irrational_pids: set[int] = set()

    # ---------- detector ----------

    def _ensure_detector(self):
        if self._detector is None:
            from app.vision.detector import Detector

            self._detector = Detector(
                model_path=self.cfg.model,
                device=self.cfg.device,
                imgsz=self.cfg.imgsz,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                human_classes=self.cfg.human_class_set(),
                threat_classes=self.cfg.threat_class_set(),
                tiles=self.cfg.tiles,
            )
        return self._detector

    # ---------- per-frame analysis ----------

    def _apply_roi(self, frame: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return frame
        h, w = frame.shape[:2]
        rx, ry, rw, rh = self.roi
        x0, y0 = int(rx * w), int(ry * h)
        x1, y1 = min(w, int((rx + rw) * w)), min(h, int((ry + rh) * h))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return frame
        return np.ascontiguousarray(frame[y0:y1, x0:x1])

    def _in_ignore(self, nx: float, ny: float) -> bool:
        return any(rx <= nx <= rx + rw and ry <= ny <= ry + rh for rx, ry, rw, rh in self.ignore)

    def _update_motion(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        scale = w / float(FLOW_W)
        small = cv2.resize(frame, (FLOW_W, max(2, int(h / scale))))
        self.gm.update(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), scale)

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int, t: float) -> dict:
        """Analyze one native frame (ROI cropping is applied here) and return
        its metadata record — same schema as the live WS packet, minus the
        JPEG. ``t`` is video time (frame_idx / fps), so runs are deterministic."""
        frame = self._apply_roi(frame_bgr)
        h, w = frame.shape[:2]
        diag = math.hypot(w, h)
        self._update_motion(frame)
        offset = self.gm.offset
        detector = self._ensure_detector()
        detections = detector.track(frame)

        self.registry.begin_frame()
        persons: list[dict] = []
        irr_now = 0
        visible = 0
        for d in detections:
            if not d.is_human or d.track_id is None:
                continue
            x0, y0, x1, y1 = d.xyxy
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if self._in_ignore(cx / w, cy / h):
                continue
            visible += 1
            stab = (cx - float(offset[0]), cy - float(offset[1]))
            hist = appearance_hist(frame, d.xyxy)
            pid = self.registry.resolve(d.track_id, t, hist, stab, diag)
            bw, bh = x1 - x0, max(y1 - y0, 1.0)
            status, prone, speed = self.behavior.update(pid, t, stab, bh, bw / bh, None)
            if status != STATUS_OK:
                irr_now += 1
                self._irrational_pids.add(pid)
            trail = self._trails.setdefault(pid, deque(maxlen=TRAIL_MAX))
            trail.append(stab)
            trail_norm = [
                [round((sx + float(offset[0])) / w, 4), round((sy + float(offset[1])) / h, 4)]
                for sx, sy in list(trail)[-24:]
            ]
            persons.append(
                {
                    "pid": pid,
                    "tid": d.track_id,
                    "box": _norm_xywh(d.xyxy, w, h),
                    "conf": round(d.conf, 2),
                    "st": status,
                    "prone": prone,
                    "sp": round(speed, 2),
                    "trail": trail_norm,
                }
            )

        self.situation.update(frame, t, None, ignore=self.ignore)
        st = self.situation.state
        return {
            "f": frame_idx,
            "t": round(t, 3),
            "wh": [w, h],
            "persons": persons,
            "threats": [],
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
            "danger": None,
            "stats": {
                "unique": self.registry.unique_total,
                "visible": visible,
                "irr_now": irr_now,
                "irr_total": len(self._irrational_pids),
                "threat": False,
            },
        }

    # ---------- full file run ----------

    def run(self, out_dir: str | Path, *, stride: int = 1, progress=None) -> dict:
        """Analyze the whole file, writing the bundle into ``out_dir``.

        ``stride`` analyzes every Nth frame (1 = every frame). ``progress`` is
        an optional callback ``(done, total)`` for status reporting. Returns
        the summary dict that is also written to meta.json.
        """
        stride = max(1, int(stride))
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(self.source_path)
        if not cap.isOpened():
            raise RuntimeError(f"Kunde inte öppna filmen: {self.source_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1.0 <= fps <= 120.0 else 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        events = EventTracker()
        analyzed = 0
        frame_idx = 0
        wh = [0, 0]
        t_start = time.monotonic()
        self._write_state(out, "running", 0, total, fps)

        with (out / "frames.jsonl").open("w", encoding="utf-8") as fh:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % stride != 0:
                    frame_idx += 1
                    continue
                t = frame_idx / fps
                meta = self.process_frame(frame, frame_idx, t)
                wh = meta["wh"]
                fh.write(json.dumps(meta, separators=(",", ":")) + "\n")
                events.feed(meta)
                analyzed += 1
                frame_idx += 1
                if analyzed % 20 == 0:
                    self._write_state(out, "running", frame_idx, total, fps)
                    if progress is not None:
                        progress(frame_idx, total)
        cap.release()

        summary = {
            "unique": self.registry.unique_total,
            "max_visible": events.max_visible,
            "still_pids": sorted(events.still_pids),
            "toward_pids": sorted(events.toward_pids),
            "fire_frames": events.fire_frames,
            "smoke_frames": events.smoke_frames,
            "n_events": len(events.events),
        }
        meta_doc = {
            "source": Path(self.source_path).name,
            "fps": round(fps, 3),
            "stride": stride,
            "frames_total": total,
            "frames_analyzed": analyzed,
            "wh": wh,
            "duration_s": round((total / fps) if total else (analyzed * stride / fps), 2),
            "model": self.cfg.model,
            "imgsz": self.cfg.imgsz,
            "conf": self.cfg.conf,
            "tiles": self.cfg.tiles,
            "created": time.time(),
            "analysis_wall_s": round(time.monotonic() - t_start, 1),
            "summary": summary,
        }
        (out / "meta.json").write_text(json.dumps(meta_doc, indent=2), encoding="utf-8")
        (out / "events.json").write_text(json.dumps({"events": events.events}, indent=2), encoding="utf-8")
        self._write_state(out, "done", frame_idx, total, fps)
        if progress is not None:
            progress(frame_idx, total)
        return meta_doc

    @staticmethod
    def _write_state(out: Path, status: str, done: int, total: int, fps: float) -> None:
        (out / "state.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "done": done,
                    "total": total,
                    "fps": round(fps, 2),
                    "pct": round(100 * done / total, 1) if total else None,
                    "updated": time.time(),
                }
            ),
            encoding="utf-8",
        )


@dataclass
class EventTracker:
    """Diffs consecutive frame records into a human-readable timeline.

    The point of the offline tool is after-action review — "did we miss
    someone? was that the right call?" — so the timeline surfaces the moments
    worth jumping to: people appearing, going STILL (possibly down), moving
    toward danger, and fire/smoke being indicated.
    """

    events: list[dict] = field(default_factory=list)
    _known_pids: set[int] = field(default_factory=set)
    _status: dict[int, str] = field(default_factory=dict)
    _fire: bool = False
    _smoke: bool = False
    max_visible: int = 0
    fire_frames: int = 0
    smoke_frames: int = 0
    still_pids: set[int] = field(default_factory=set)
    toward_pids: set[int] = field(default_factory=set)

    def _add(self, meta: dict, kind: str, text: str, sev: str, pid: int | None = None) -> None:
        self.events.append(
            {"f": meta["f"], "t": meta["t"], "kind": kind, "text": text, "sev": sev, "pid": pid}
        )

    def feed(self, meta: dict) -> None:
        self.max_visible = max(self.max_visible, meta["stats"]["visible"])
        live_pids = set()
        for p in meta["persons"]:
            pid, stt = p["pid"], p["st"]
            live_pids.add(pid)
            if pid not in self._known_pids:
                self._known_pids.add(pid)
                self._add(meta, "appear", f"P{pid} upptäckt", "info", pid)
            prev = self._status.get(pid, STATUS_OK)
            if stt != prev:
                if stt == "still":
                    self.still_pids.add(pid)
                    txt = f"P{pid} STILLA" + (" – LIGGER" if p.get("prone") else "")
                    self._add(meta, "still", txt, "alert", pid)
                elif stt == "toward_danger":
                    self.toward_pids.add(pid)
                    self._add(meta, "toward", f"P{pid} rör sig MOT FARA", "warn", pid)
                elif prev != STATUS_OK:
                    self._add(meta, "normal", f"P{pid} åter normal", "info", pid)
                self._status[pid] = stt
        # forget people no longer visible so a reappearance re-fires "upptäckt"
        for pid in list(self._status):
            if pid not in live_pids:
                self._status.pop(pid, None)

        hz = meta["hazards"]
        fire = hz["fire"] is not None
        smoke = hz["smoke"] is not None
        if fire:
            self.fire_frames += 1
        if smoke:
            self.smoke_frames += 1
        if fire and not self._fire:
            self._add(meta, "fire", "Brand indikerad", "alert")
        elif not fire and self._fire:
            self._add(meta, "fire_end", "Brand ej längre indikerad", "info")
        if smoke and not self._smoke:
            self._add(meta, "smoke", "Rök indikerad", "warn")
        elif not smoke and self._smoke:
            self._add(meta, "smoke_end", "Rök ej längre indikerad", "info")
        self._fire, self._smoke = fire, smoke
