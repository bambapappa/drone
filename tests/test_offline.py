"""Offline analyzer + event timeline — ML-free via a stub detector."""

from __future__ import annotations

import numpy as np

from app.core.config import Settings
from app.vision.detector import Detection
from app.vision.offline import EventTracker, OfflineAnalyzer


class StubDetector:
    """Returns a scripted list of Detection per call (last list repeats)."""

    def __init__(self, scripted: list[list[Detection]]):
        self.scripted = scripted
        self.names = {0: "person"}
        self._i = 0

    def track(self, frame):
        dets = self.scripted[min(self._i, len(self.scripted) - 1)]
        self._i += 1
        return dets


def _person(tid, xyxy, conf=0.5):
    return Detection(track_id=tid, cls_name="person", conf=conf, xyxy=xyxy, is_human=True, is_threat=False)


def _frame():
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_process_frame_schema_and_counts():
    cfg = Settings()
    det = StubDetector([[_person(1, (40, 40, 90, 170)), _person(2, (200, 50, 250, 180))]])
    an = OfflineAnalyzer(cfg, "x.mp4", detector=det)
    meta = an.process_frame(_frame(), frame_idx=0, t=0.0)

    assert set(meta) >= {"f", "t", "wh", "persons", "hazards", "base", "danger", "stats"}
    assert meta["f"] == 0 and meta["wh"] == [320, 240]
    assert len(meta["persons"]) == 2
    assert meta["stats"]["visible"] == 2
    for p in meta["persons"]:
        assert {"pid", "box", "conf", "st", "trail"} <= set(p)
        assert len(p["box"]) == 4 and all(0 <= v <= 1.2 for v in p["box"])
        assert p["st"] == "ok"  # no history yet


def test_ignore_region_filters_detection():
    cfg = Settings(ignore_regions="0.0,0.0,0.4,1.0")  # left 40% excluded
    det = StubDetector([[_person(1, (10, 40, 60, 170)), _person(2, (200, 50, 250, 180))]])
    an = OfflineAnalyzer(cfg, "x.mp4", detector=det)
    meta = an.process_frame(_frame(), 0, 0.0)
    assert meta["stats"]["visible"] == 1  # left-side person dropped
    assert meta["persons"][0]["box"][0] > 0.4


def test_stable_pid_across_frames():
    cfg = Settings()
    det = StubDetector([[_person(7, (40, 40, 90, 170))]])  # same track id repeats
    an = OfflineAnalyzer(cfg, "x.mp4", detector=det)
    pids = {an.process_frame(_frame(), i, i / 25.0)["persons"][0]["pid"] for i in range(5)}
    assert pids == {1}  # one stable person


def test_event_tracker_transitions():
    ev = EventTracker()
    base = {
        "f": 0,
        "t": 0.0,
        "stats": {"visible": 1},
        "hazards": {"fire": None, "smoke": None},
        "persons": [{"pid": 1, "st": "ok", "prone": False}],
    }
    ev.feed(base)
    kinds = [e["kind"] for e in ev.events]
    assert "appear" in kinds  # P1 discovered

    still = {**base, "f": 100, "t": 4.0, "persons": [{"pid": 1, "st": "still", "prone": True}]}
    ev.feed(still)
    last = ev.events[-1]
    assert last["kind"] == "still" and "LIGGER" in last["text"] and 1 in ev.still_pids


def test_event_tracker_fire_smoke_edges():
    ev = EventTracker()
    no_hz = {
        "f": 0,
        "t": 0.0,
        "stats": {"visible": 0},
        "persons": [],
        "hazards": {"fire": None, "smoke": None},
    }
    on_hz = {
        "f": 1,
        "t": 0.1,
        "stats": {"visible": 0},
        "persons": [],
        "hazards": {
            "fire": {"pos": [0.5, 0.5], "area": 0.02},
            "smoke": {"pos": [0.4, 0.4], "drift": [0, 0]},
        },
    }
    ev.feed(no_hz)
    ev.feed(on_hz)
    ev.feed(no_hz)
    kinds = [e["kind"] for e in ev.events]
    assert kinds.count("fire") == 1 and kinds.count("smoke") == 1
    assert "fire_end" in kinds and "smoke_end" in kinds
    assert ev.fire_frames == 1 and ev.smoke_frames == 1
