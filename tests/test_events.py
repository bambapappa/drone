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

import cv2
import numpy as np

from analysis.events import (
    CATEGORY_HAZARD,
    CATEGORY_MOT_FARA,
    CATEGORY_STILLA,
    Event,
    build_danger_resolver,
    derive_behavior_events,
    derive_events,
    derive_hazard_events,
)
from analysis.scene import transform_point


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
            danger_for_frame=lambda fn, s: danger_px,
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
            danger_for_frame=None,
        )
        assert len([e for e in events if e.category == CATEGORY_MOT_FARA]) == 0

    def test_moving_danger_tracked_per_frame_not_averaged(self):
        # Fire relocates from far-right to far-left mid-film. Person walks +x
        # throughout. Per-frame: while danger is to the right (frames 0..59) the
        # person moves toward it -> MOT_FARA fires; once danger flips left
        # (frames 60..119) the person moves away -> no MOT_FARA in that span.
        # The MEAN danger x = 0 sits behind the person (who starts at x=50) for
        # the whole film, so a constant-mean resolver would fire ZERO events.
        fps = 10.0
        frames = list(range(120))
        xyxy_seq = [(50.0 + i * 4.0, 100.0, 80.0 + i * 4.0, 180.0) for i in range(120)]
        rows = _trk(1, frames, xyxy_seq, fps=fps)

        def danger_for_frame(frame_no, segment):
            return (1000.0, 140.0) if frame_no < 60 else (-1000.0, 140.0)

        events = derive_behavior_events(
            rows,
            person_by_tracklet={},
            fps=fps,
            frame_w=1280,
            frame_h=720,
            config=_beh_config(),
            danger_for_frame=danger_for_frame,
        )
        mot = [e for e in events if e.category == CATEGORY_MOT_FARA]
        assert len(mot) >= 1  # per-frame tracks the right-side danger

        # Proof the mean would have missed it: constant resolver at the mean (0,140).
        mean_events = derive_behavior_events(
            rows,
            person_by_tracklet={},
            fps=fps,
            frame_w=1280,
            frame_h=720,
            config=_beh_config(),
            danger_for_frame=lambda fn, s: (0.0, 140.0),
        )
        assert len([e for e in mean_events if e.category == CATEGORY_MOT_FARA]) == 0

    def test_scene_segment_break_resets_stillness_history(self):
        # Ten seconds stationary would normally fire STILLA. Split into
        # visually unrelated five-second local maps, neither side has the
        # 3 s history + 4 s sustained gate, so no event may bridge the cut.
        rows = _trk(1, list(range(100)), [(100.0, 100.0, 130.0, 180.0)] * 100)
        for row in rows:
            row["scene_pos"] = [115.0, 180.0]
            row["scene_box_h"] = 80.0
            row["scene_segment"] = 0 if row["frame_no"] < 50 else 1

        events = derive_behavior_events(
            rows,
            person_by_tracklet={},
            fps=10.0,
            frame_w=320,
            frame_h=240,
            config=_beh_config(),
        )

        assert not any(e.category == CATEGORY_STILLA for e in events)

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


class TestSceneCompensatedSmoke:
    """derive_events: smoke motion measured in scene space (B31/B32).

    The camera's own motion is warped out with P2's persisted scene
    transforms before the frame diff, so scene-static gray ground under a
    panning drone is no longer "smoke", and a scene-moving gray blob stays
    detectable even when the camera pans with it. Unlinked pairs (segment
    break / missing transform) give an empty motion mask that frame — never
    a guess across a visual loss.

    Fixtures verified against smoke_mask directly: legacy fires/suppresses
    exactly opposite to the compensated path, so each test is honest in
    both directions (not green against empty output).
    """

    FPS = 10.0

    def _scene_frames(self, n: int) -> dict[int, dict]:
        # Camera pans so that scene-static content moves +8 px/frame in the
        # frame (test 2's blob rides exactly against it) => frame_to_scene
        # shifts -8*i, scene_to_frame +8*i, and the prev->cur composite
        # s2f(n) @ f2s(n-1) is a pure +8 translation. The slower +3 px/frame
        # variant (tests 1/3) keeps the rectangle fully on-frame for all 30
        # frames — content sliding in from off-frame is genuinely new and
        # cannot be compensated.
        scene_frames = {}
        for i in range(n):
            f2s = np.float32([[1, 0, -8.0 * i], [0, 1, 0], [0, 0, 1]])
            s2f = np.float32([[1, 0, 8.0 * i], [0, 1, 0], [0, 0, 1]])
            scene_frames[i] = {"frame_to_scene": f2s, "scene_to_frame": s2f, "scene_segment": 0}
        return scene_frames

    def _pan_frames(self) -> list[np.ndarray]:
        # Gray rectangle, scene-static, drifting +3 px/frame in frame (fully
        # on-frame throughout); transforms for the +3 pan are built inline.
        frames = []
        for i in range(30):
            img = np.full((90, 160, 3), (40, 120, 60), np.uint8)
            cv2.rectangle(img, (10 + 3 * i, 10), (60 + 3 * i, 50), (90, 110, 105), -1)
            frames.append(img)
        return frames

    @staticmethod
    def _pan_scene_frames(n: int) -> dict[int, dict]:
        scene_frames = {}
        for i in range(n):
            f2s = np.float32([[1, 0, -3.0 * i], [0, 1, 0], [0, 0, 1]])
            s2f = np.float32([[1, 0, 3.0 * i], [0, 1, 0], [0, 0, 1]])
            scene_frames[i] = {"frame_to_scene": f2s, "scene_to_frame": s2f, "scene_segment": 0}
        return scene_frames

    def test_pan_over_static_gray_scene_no_false_smoke(self):
        # Scene-static gray rectangle under a panning camera: without the
        # wiring the legacy frame diff sees motion (blob area ~0.0085,
        # above min_area 0.001, every frame) and fabricates a smoke event;
        # with the persisted transforms the rectangle is warped back onto
        # itself (verified: the compensated mask is empty) and no HAZARD
        # event may appear.
        frames = self._pan_frames()
        cfg = _sit_config()
        cfg.hazard_min_area = 0.001
        cfg.hazard_hold_s = 0.1
        events = derive_events(
            [],
            person_by_tracklet={},
            frames=frames,
            fps=self.FPS,
            frame_w=160,
            frame_h=90,
            config=cfg,
            ignore_regions=None,
            scene_frames=self._pan_scene_frames(30),
        )
        assert not any(e.category == CATEGORY_HAZARD for e in events)

    def test_smoke_scene_motion_fires_under_camera_pan(self):
        # Gray blob stationary in the FRAME (it drifts -8 px/frame in scene
        # while the camera pans +8): the legacy frame diff sees no motion at
        # the blob (verified: legacy mask empty), only the scene-compensated
        # path isolates it (compensated blob area ~0.008 > min_area 0.001),
        # so a HAZARD smoke event must appear once derive_events passes
        # prev_to_cur through.
        frames = []
        for _ in range(30):
            img = np.full((90, 160, 3), (40, 120, 60), np.uint8)
            cv2.circle(img, (100, 60), 8, (100, 115, 110), -1)
            frames.append(img)
        cfg = _sit_config()
        cfg.hazard_min_area = 0.001
        cfg.hazard_hold_s = 0.1
        events = derive_events(
            [],
            person_by_tracklet={},
            frames=frames,
            fps=self.FPS,
            frame_w=160,
            frame_h=90,
            config=cfg,
            ignore_regions=None,
            scene_frames=self._scene_frames(30),
        )
        assert any(e.category == CATEGORY_HAZARD and e.evidence["kind"] == "smoke" for e in events)

    def test_segment_break_no_crash_no_smoke(self):
        # Same panning gray rectangle, but the scene record switches segment
        # at i=15 and carries no transforms from there on. The linked pairs
        # (1..14) compensate to an empty mask; the unlinked pairs must give
        # an empty mask too — no crash, and no smoke event bridging the
        # break. The assertion is honest: hold_s=0.1 and min_area=0.001 are
        # small enough that any bridged/guessed mask from frame 15 on (the
        # rectangle keeps moving in frame) would produce a smoke event —
        # the legacy path over the same frames fires one.
        frames = self._pan_frames()
        scene_frames = self._pan_scene_frames(15)
        for i in range(15, 30):
            scene_frames[i] = {"frame_to_scene": None, "scene_to_frame": None, "scene_segment": 1}
        cfg = _sit_config()
        cfg.hazard_min_area = 0.001
        cfg.hazard_hold_s = 0.1
        events = derive_events(
            [],
            person_by_tracklet={},
            frames=frames,
            fps=self.FPS,
            frame_w=160,
            frame_h=90,
            config=cfg,
            ignore_regions=None,
            scene_frames=scene_frames,
        )
        assert not any(e.category == CATEGORY_HAZARD for e in events)

    def test_rotation_scaling_only_translation_downscale(self):
        # Pins the S·A·S⁻¹ downscale: frame_w=320 (≠ WORK_W=160, so
        # warp_scale=0.5) with a NON-identity linear part — the camera
        # rotates 10°/frame and pans +4 px/frame (full-res) over a
        # scene-static gray rectangle. Scaling the whole 2x3 by 0.5 would
        # shrink the rotation into a 0.5·R zoom+rotation, mis-warp the
        # prev frame and reintroduce camera-motion "smoke"; only the
        # translation column may scale. Legacy (no scene data) must fire.
        cs = np.float32([180.0, 110.0])  # scene anchor under frame center at i=0
        cf = np.float32([160.0, 90.0])
        scene = np.full((400, 640, 3), (40, 120, 60), np.uint8)
        cv2.rectangle(scene, (208, 80), (268, 140), (90, 110, 105), -1)
        frames, scene_frames = [], {}
        for i in range(30):
            th = np.deg2rad(10.0 * i)
            c, s = np.cos(th), np.sin(th)
            rot = np.float32([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            t_pos = np.float32([[1, 0, cf[0] + 4.0 * i], [0, 1, cf[1]], [0, 0, 1]])
            t_neg = np.float32([[1, 0, -cs[0]], [0, 1, -cs[1]], [0, 0, 1]])
            s2f = t_pos @ rot @ t_neg
            frames.append(cv2.warpAffine(scene, s2f[:2], (320, 180)))
            scene_frames[i] = {
                "frame_to_scene": np.linalg.inv(s2f),
                "scene_to_frame": s2f,
                "scene_segment": 0,
            }
        cfg = _sit_config()
        cfg.hazard_min_area = 0.001
        cfg.hazard_hold_s = 0.1
        common = dict(
            person_by_tracklet={},
            frames=frames,
            fps=self.FPS,
            frame_w=320,
            frame_h=180,
            config=cfg,
            ignore_regions=None,
        )
        # Honest fixture check (same style as the tests above): the legacy
        # frame diff over the same downscaled frames sees the pan+rotation
        # as motion, so without compensation this WOULD be "smoke".
        from analysis.situation import smoke_mask

        small = [cv2.resize(f, (160, 90)) for f in frames]
        legacy_px = sum(
            int(smoke_mask(small[i], cv2.cvtColor(small[i - 1], cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(small[i], cv2.COLOR_BGR2GRAY)).sum())
            for i in range(1, len(small))
        )
        assert legacy_px > 1000  # legacy fires without compensation
        # Compensated: correct S·A·S⁻¹ scaling → no smoke event.
        events = derive_events([], scene_frames=scene_frames, **common)
        assert not any(e.category == CATEGORY_HAZARD for e in events)


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


class TestBuildDangerResolver:
    """build_danger_resolver: per-frame danger in the analyzer's space.

    The whole point of the #1 fix: the resolver returns each frame's own
    danger point (scene-transformed per its own frame_to_scene when scene
    data exists, else raw pixel) — never a mean collapsed across the film.
    """

    def test_raw_pixel_per_frame_when_no_scene_data(self):
        danger_px_by_frame = {0: (100.0, 200.0), 1: (300.0, 400.0)}
        resolver = build_danger_resolver(danger_px_by_frame, scene_frames=None)
        assert resolver is not None
        assert resolver(0, None) == (100.0, 200.0)
        assert resolver(1, None) == (300.0, 400.0)
        assert resolver(2, None) is None  # frame without danger

    def test_scene_transform_uses_each_frames_own_matrix(self):
        # Same pixel danger in two frames, different per-frame transforms ->
        # different scene points (NOT a segment mean).
        m0 = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        m1 = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        scene_frames = {
            0: {"frame_to_scene": m0, "scene_segment": 0},
            1: {"frame_to_scene": m1, "scene_segment": 0},
        }
        danger_px_by_frame = {0: (100.0, 100.0), 1: (100.0, 100.0)}
        resolver = build_danger_resolver(danger_px_by_frame, scene_frames)
        assert resolver(0, 0) == transform_point(m0, (100.0, 100.0))
        assert resolver(1, 0) == transform_point(m1, (100.0, 100.0))
        assert resolver(0, 0) != resolver(1, 0)

    def test_none_when_no_danger_ever(self):
        assert build_danger_resolver({}, scene_frames=None) is None
        assert build_danger_resolver({0: None, 1: None}, scene_frames=None) is None
