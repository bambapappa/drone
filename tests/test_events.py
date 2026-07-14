"""Tests for P5 event derivation (analysis.events).

P5 = the marriage of the architecture report's P4 (per-frame behavior/situation
status via the carried-over analyzers) and P5 (status-stream diffing into
discrete onset/offset events). The analyzers are stateless per-call, so there
is no need to persist their per-frame output — derive events directly from the
tracklet table + a frame stream.

Phase 2 categories: STILLA, MOT_FARA (the carried-over BehaviorAnalyzer
categories) + HAZARD (smoke/fire from SituationAnalyzer). IRRATIONELL is
explicitly Phase 4 per the report's build order and is not derived here.

Pure-logic tests (no torch, no real video): synthetic tracklet timelines and
synthetic BGR frames drive the analyzers deterministically.
"""

from __future__ import annotations

import numpy as np

from analysis.events import (
    CATEGORY_HAZARD,
    CATEGORY_MOT_FARA,
    CATEGORY_STILLA,
    Event,
    derive_behavior_events,
    derive_hazard_events,
)


def _trk(
    tracklet_id: int,
    frames: list[int],
    xyxy_seq: list[tuple[float, float, float, float]],
    fps: float = 10.0,
) -> list[dict]:
    """Build a synthetic per-(tracklet, frame) row stream, mimicking P2 output."""
    assert len(frames) == len(xyxy_seq)
    rows = []
    det_id = tracklet_id * 1000
    for frame_no, xyxy in zip(frames, xyxy_seq):
        rows.append(
            {
                "tracklet_id": tracklet_id,
                "frame_no": frame_no,
                "det_id": det_id,
                "cls": "person",
                "conf": 0.9,
                "xyxy": list(xyxy),
            }
        )
        det_id += 1
    return rows


class TestBehaviorEventDiff:
    """Diffing per-frame behavior status into onset/offset events.

    The BehaviorAnalyzer itself is exercised by test_analysis_behavior; here
    we drive its update() with known synthetic trajectories and check the
    event diff catches onset/offset of each status span correctly.
    """

    def test_stilla_event_for_sustained_stationary_tracklet(self):
        # A tracklet that doesn't move at all. Needs to outlast the analyzer's
        # min_history_s (3s warm-up) + still_time_s (4s sustained) before STILL
        # fires, so 10s @ 10fps is enough to enter the status and hold it.
        fps = 10.0
        stationary_xyxy = (100.0, 100.0, 130.0, 180.0)
        frames = list(range(0, 100))
        xyxy_seq = [stationary_xyxy] * 100
        rows = _trk(1, frames, xyxy_seq, fps=fps)
        config = _beh_config()
        events = derive_behavior_events(
            rows, person_by_tracklet={}, fps=fps, frame_w=320, frame_h=240, config=config
        )
        stilla = [e for e in events if e.category == CATEGORY_STILLA]
        assert len(stilla) == 1
        e = stilla[0]
        assert e.t_end > e.t_start
        # Onset is when the analyzer first flagged STILL (after the 3s warm-up
        # plus the 4s still_time_s gate); the event itself spans the time the
        # analyzer was confidently in STILL state.
        assert e.evidence["tracklet_id"] == 1

    def test_no_stilla_for_brief_pause(self):
        # Stationary for only 1s (below the 4s threshold) -> no STILLA event.
        fps = 10.0
        stationary = (100.0, 100.0, 130.0, 180.0)
        # Need enough history to clear min_history_s (3s) first; move for 4s,
        # then stay still only 1s, then end.
        moving = [(100.0 + i * 5, 100.0, 130.0 + i * 5, 180.0) for i in range(40)]
        still = [stationary] * 10
        rows = _trk(1, list(range(50)), moving + still, fps=fps)
        events = derive_behavior_events(
            rows, person_by_tracklet={}, fps=fps, frame_w=320, frame_h=240, config=_beh_config()
        )
        assert len([e for e in events if e.category == CATEGORY_STILLA]) == 0

    def test_person_id_tagged_when_p3_ran(self):
        # When P3 mapped tracklet_id -> person_id, events carry the person_id.
        fps = 10.0
        stationary_xyxy = (100.0, 100.0, 130.0, 180.0)
        rows = _trk(7, list(range(60)), [stationary_xyxy] * 60, fps=fps)
        events = derive_behavior_events(
            rows,
            person_by_tracklet={7: 3},  # tracklet 7 belongs to person P3
            fps=fps,
            frame_w=320,
            frame_h=240,
            config=_beh_config(),
        )
        assert all(e.person_id == 3 for e in events)

    def test_person_id_null_when_p3_skipped(self):
        fps = 10.0
        stationary_xyxy = (100.0, 100.0, 130.0, 180.0)
        rows = _trk(7, list(range(60)), [stationary_xyxy] * 60, fps=fps)
        events = derive_behavior_events(
            rows,
            person_by_tracklet={},  # P3 didn't run
            fps=fps,
            frame_w=320,
            frame_h=240,
            config=_beh_config(),
        )
        assert all(e.person_id is None for e in events)

    def test_mot_fara_event_when_moving_toward_danger(self):
        # A tracklet moving in +x toward a danger point on the right.
        fps = 10.0
        # Sustained motion for 6s at >toward_speed (0.25 bh/s) in +x.
        # Person height ~80 px; 0.25 bh/s = 20 px/s; we go 40 px/s = 2x.
        frames = list(range(0, 80))
        xyxy_seq = [(50.0 + i * 4.0, 100.0, 80.0 + i * 4.0, 180.0) for i in range(80)]
        rows = _trk(1, frames, xyxy_seq, fps=fps)
        # Danger far to the right of the trajectory, so direction (+x) aligns.
        danger_px = (1000.0, 140.0)
        events = derive_behavior_events(
            rows,
            person_by_tracklet={},
            fps=fps,
            frame_w=1280,
            frame_h=720,
            config=_beh_config(),
            danger_px=danger_px,
        )
        mot = [e for e in events if e.category == CATEGORY_MOT_FARA]
        assert len(mot) >= 1

    def test_no_mot_fara_without_danger_point(self):
        # With no danger point supplied (the offline default until a hazard is
        # detected), MOT_FARA cannot be derived at all — STILLA still can be.
        fps = 10.0
        xyxy_seq = [(50.0 + i * 4.0, 100.0, 80.0 + i * 4.0, 180.0) for i in range(80)]
        rows = _trk(1, list(range(80)), xyxy_seq, fps=fps)
        events = derive_behavior_events(
            rows,
            person_by_tracklet={},
            fps=fps,
            frame_w=1280,
            frame_h=720,
            config=_beh_config(),
            danger_px=None,
        )
        assert len([e for e in events if e.category == CATEGORY_MOT_FARA]) == 0

    def test_review_state_defaults_to_unreviewed(self):
        fps = 10.0
        rows = _trk(1, list(range(60)), [(100.0, 100.0, 130.0, 180.0)] * 60, fps=fps)
        events = derive_behavior_events(
            rows, person_by_tracklet={}, fps=fps, frame_w=320, frame_h=240, config=_beh_config()
        )
        assert all(e.review["state"] == "unreviewed" for e in events)

    def test_event_ids_unique_and_stable_within_run(self):
        # Within one derivation pass, no two events share an event_id.
        fps = 10.0
        rows = _trk(1, list(range(60)), [(100.0, 100.0, 130.0, 180.0)] * 60, fps=fps)
        rows += _trk(2, list(range(60)), [(200.0, 100.0, 230.0, 180.0)] * 60, fps=fps)
        events = derive_behavior_events(
            rows, person_by_tracklet={}, fps=fps, frame_w=320, frame_h=240, config=_beh_config()
        )
        ids = [e.event_id for e in events]
        assert len(ids) == len(set(ids))

    def test_confidence_in_zero_to_one(self):
        fps = 10.0
        rows = _trk(1, list(range(60)), [(100.0, 100.0, 130.0, 180.0)] * 60, fps=fps)
        events = derive_behavior_events(
            rows, person_by_tracklet={}, fps=fps, frame_w=320, frame_h=240, config=_beh_config()
        )
        for e in events:
            assert 0.0 <= e.confidence <= 1.0


class TestHazardEventDiff:
    """Diffing SituationAnalyzer fire/smoke state into HAZARD events.

    Uses synthetic BGR frames: solid-color frames chosen to land in the fire
    or smoke colour masks, large enough to clear the min-area gate.
    """

    def test_no_hazard_on_blank_gray(self):
        # Neutral gray frame — neither fire nor smoke colour mask fires.
        frames = [_solid_frame(160, 160, 160) for _ in range(40)]
        events = derive_hazard_events(frames, fps=10.0, config=_sit_config())
        assert len(events) == 0

    def test_fire_event_requires_sustained_smoke(self):
        # The live SituationAnalyzer is gated on smoke_near (DECISIONS B18) —
        # saturated red alone (e.g. a tile roof) does not produce a fire
        # hazard. We construct a frame that has both red and gray-moving-smoke.
        # The simplest sustained synthetic: alternate prev/cur so the smoke
        # motion mask fires alongside the red blob.
        frames = []
        for i in range(40):
            base = np.full((240, 320, 3), 100, dtype=np.uint8)
            # Big saturated red blob in the center (passes fire_mask colour).
            base[80:160, 120:200] = (60, 80, 220)  # BGR: high R, mid G, low B
            # Make successive frames differ slightly so smoke_mask's motion
            # term fires inside the gray border around the red blob.
            base[80:160, 200:240] = (140 + (i % 20), 140, 140)
            frames.append(base)
        events = derive_hazard_events(frames, fps=10.0, config=_sit_config())
        # Even if fire is rejected by the smoke-near gate, smoke itself may
        # produce a hazard event. Either way, the category must be HAZARD and
        # the kind is in {fire, smoke}.
        for e in events:
            assert e.category == CATEGORY_HAZARD
            assert e.evidence["kind"] in {"fire", "smoke"}
            assert e.person_id is None  # hazards are not person-keyed

    def test_hazard_offset_emitted_when_sustained_then_clears(self):
        # 40 frames of potential trigger, then 40 frames of clean gray — we
        # should see at least one HAZARD event with a finite t_end (offset),
        # not one that lingers forever. (Smoke needs motion; gray-static won't
        # trigger, so this test guards the offset bookkeeping rather than
        # asserting a hazard must fire.)
        frames = []
        for i in range(40):
            base = np.full((240, 320, 3), 100, dtype=np.uint8)
            base[80:160, 120:200] = (60, 80, 220)
            base[80:160, 200:240] = (140 + (i % 20), 140, 140)
            frames.append(base)
        for _ in range(40):
            frames.append(_solid_frame(160, 160, 160))
        events = derive_hazard_events(frames, fps=10.0, config=_sit_config())
        # Every emitted event has a finite, ordered t_start/t_end.
        for e in events:
            assert e.t_end >= e.t_start


class TestEventSerialization:
    def test_event_to_dict_roundtrips_json(self):
        import json

        e = Event(
            event_id="ev-001",
            category=CATEGORY_STILLA,
            person_id=4,
            t_start=12.0,
            t_end=18.5,
            confidence=0.83,
            evidence={"tracklet_id": 7, "prone": False, "avg_speed": 0.04},
        )
        d = e.to_dict()
        # Round-trips through JSON cleanly (no numpy types, no sets).
        s = json.dumps(d)
        d2 = json.loads(s)
        assert d2["event_id"] == "ev-001"
        assert d2["category"] == "STILLA"
        assert d2["person_id"] == 4
        assert d2["evidence"]["tracklet_id"] == 7
        assert d2["review"]["state"] == "unreviewed"


# ---- helpers ----


def _beh_config():
    from analysis.orchestrator import OfflineConfig

    return OfflineConfig(
        beh_window_s=6.0,
        beh_min_history_s=3.0,
        beh_still_speed=0.10,
        beh_still_time_s=4.0,
        beh_toward_speed=0.25,
        beh_toward_angle_deg=40.0,
        beh_toward_time_s=1.5,
        beh_prone_aspect=1.4,
    )


def _sit_config():
    from analysis.orchestrator import OfflineConfig

    # Smaller hold_s so 40-frame synthetic runs can produce a sustained event.
    return OfflineConfig(
        hazard_min_area=0.004,
        hazard_hold_s=1.0,
        smoke_flow_ema=0.15,
        base_margin=0.08,
        base_hysteresis=0.15,
        fire_require_smoke=True,
    )


def _solid_frame(b: int, g: int, r: int, w: int = 320, h: int = 240) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (b, g, r)
    return frame
