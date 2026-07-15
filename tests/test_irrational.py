"""Tests for P4 IRRATIONELL behavior (analysis.irrational).

One constructed-trajectory test per sub-signal (should and should-not
trigger), an ensemble combination test, the STILLA-wins-over-IRRATIONELL
precedence rule, and a determinism test — same discipline as
test_events.py's STILLA/MOT_FARA coverage.

Pure-logic tests: synthetic tracklet timelines, no torch/video/ML.
"""

from __future__ import annotations

import math

from analysis.irrational import (
    CATEGORY_IRRATIONELL,
    IrrationalConfig,
    derive_irrational_events,
)

FPS = 10.0
BOX_W = 30.0
BOX_H = 80.0  # body height in px; 1 bh/s = 80 px/s at this box size


def _trk(
    tracklet_id: int, xyxy_seq: list[tuple[float, float, float, float]], frame_start: int = 0
) -> list[dict]:
    """Build synthetic per-(tracklet, frame) rows, mimicking P2 output —
    same shape as test_events.py's _trk helper."""
    rows = []
    det_id = tracklet_id * 100_000
    for i, xyxy in enumerate(xyxy_seq):
        rows.append(
            {
                "tracklet_id": tracklet_id,
                "frame_no": frame_start + i,
                "det_id": det_id,
                "cls": "person",
                "conf": 0.9,
                "xyxy": list(xyxy),
            }
        )
        det_id += 1
    return rows


def _box_at(cx: float, cy: float, w: float = BOX_W, h: float = BOX_H) -> tuple[float, float, float, float]:
    """Box whose foot-center ((x0+x1)/2, y1) is (cx, cy)."""
    return (cx - w / 2, cy - h, cx + w / 2, cy)


def _stationary(n: int, cx: float, cy: float) -> list[tuple[float, float, float, float]]:
    return [_box_at(cx, cy) for _ in range(n)]


def _cfg(**overrides) -> IrrationalConfig:
    return IrrationalConfig(**overrides)


class TestErraticPath:
    def test_small_circling_fires_erratic(self):
        # Small-radius circling: high tortuosity + heading spread, but the
        # excursion from any pivot never approaches oscillation's threshold
        # and there's no second tracklet, so only "erratic" can fire.
        n = 150
        radius = 15.0
        step = 1.5  # radians/frame
        seq = []
        for i in range(n):
            ang = i * step
            cx = 300.0 + radius * math.cos(ang)
            cy = 300.0 + radius * math.sin(ang)
            seq.append(_box_at(cx, cy))
        rows = _trk(1, seq)
        events = derive_irrational_events(rows, person_by_tracklet={}, fps=FPS, config=_cfg())
        irr = [e for e in events if e.category == CATEGORY_IRRATIONELL]
        assert len(irr) >= 1
        assert "erratic" in irr[0].evidence["sub_signals"]
        assert irr[0].evidence["sub_signals"]["erratic"]["tortuosity"] > 3.0

    def test_straight_line_does_not_fire_erratic(self):
        n = 150
        seq = [_box_at(100.0 + i * 5.0, 300.0) for i in range(n)]
        rows = _trk(1, seq)
        events = derive_irrational_events(rows, person_by_tracklet={}, fps=FPS, config=_cfg())
        irr = [e for e in events if e.category == CATEGORY_IRRATIONELL]
        assert not any("erratic" in e.evidence["sub_signals"] for e in irr)


class TestPanicSprint:
    def test_lone_sprinter_among_stationary_group_fires_sprint(self):
        # Subject sprints in +x while two others stand still nearby —
        # group-relative, so the near-zero group median makes any sustained
        # fast motion qualify (report §4: group-relative avoids flagging a
        # general evacuation, which this test doesn't exercise — see the
        # negative case below).
        n = 100
        subject = [_box_at(100.0 + i * 40.0, 300.0) for i in range(n)]  # 400 px/s = 5 bh/s
        bystander_a = _stationary(n, 120.0, 320.0)
        bystander_b = _stationary(n, 140.0, 340.0)
        rows = _trk(1, subject) + _trk(2, bystander_a) + _trk(3, bystander_b)
        events = derive_irrational_events(
            rows, person_by_tracklet={}, fps=FPS, config=_cfg(erratic_tortuosity=1e9)
        )
        subject_irr = [
            e for e in events if e.category == CATEGORY_IRRATIONELL and e.evidence["tracklet_id"] == 1
        ]
        assert len(subject_irr) >= 1
        assert "sprint" in subject_irr[0].evidence["sub_signals"]

    def test_general_evacuation_does_not_flag_everyone(self):
        # Everyone runs at the same speed — group-relative gate means no one
        # is > 2x the group median, so sprint must not fire for any of them.
        n = 100
        a = [_box_at(100.0 + i * 40.0, 300.0) for i in range(n)]
        b = [_box_at(100.0 + i * 40.0, 340.0) for i in range(n)]
        c = [_box_at(100.0 + i * 40.0, 380.0) for i in range(n)]
        rows = _trk(1, a) + _trk(2, b) + _trk(3, c)
        events = derive_irrational_events(rows, person_by_tracklet={}, fps=FPS, config=_cfg())
        assert not any(
            "sprint" in e.evidence["sub_signals"] for e in events if e.category == CATEGORY_IRRATIONELL
        )


class TestCounterFlow:
    def test_lone_against_coherent_group_fires_counterflow(self):
        # Three group members move steadily in +x (dominant flow ~0 deg);
        # the subject moves steadily in -x (~180 deg off) nearby.
        n = 120
        subject = [_box_at(500.0 - i * 5.0, 300.0) for i in range(n)]
        member_a = [_box_at(300.0 + i * 5.0, 260.0) for i in range(n)]
        member_b = [_box_at(300.0 + i * 5.0, 300.0) for i in range(n)]
        member_c = [_box_at(300.0 + i * 5.0, 340.0) for i in range(n)]
        rows = _trk(1, subject) + _trk(2, member_a) + _trk(3, member_b) + _trk(4, member_c)
        events = derive_irrational_events(
            rows, person_by_tracklet={}, fps=FPS, config=_cfg(erratic_tortuosity=1e9)
        )
        subject_irr = [
            e for e in events if e.category == CATEGORY_IRRATIONELL and e.evidence["tracklet_id"] == 1
        ]
        assert len(subject_irr) >= 1
        assert "counterflow" in subject_irr[0].evidence["sub_signals"]
        assert subject_irr[0].evidence["sub_signals"]["counterflow"]["angle_deg"] > 120.0

    def test_no_counterflow_with_too_few_neighbors(self):
        # Only one other tracklet (below counterflow_min_neighbors=3) moving
        # the opposite way — must not fire counter-flow regardless of angle.
        n = 120
        subject = [_box_at(500.0 - i * 5.0, 300.0) for i in range(n)]
        member_a = [_box_at(300.0 + i * 5.0, 300.0) for i in range(n)]
        rows = _trk(1, subject) + _trk(2, member_a)
        events = derive_irrational_events(rows, person_by_tracklet={}, fps=FPS, config=_cfg())
        assert not any(
            "counterflow" in e.evidence["sub_signals"] for e in events if e.category == CATEGORY_IRRATIONELL
        )


class TestOscillation:
    def test_zigzag_between_two_points_fires_oscillation(self):
        # 5 legs of 2s each (20 frames @10fps), amplitude 200px (2.5 bh) —
        # well above oscillation_min_excursion_bh (1.5 bh) — produces 4
        # reversals inside the 30s window.
        leg_frames = 20
        legs = 5
        amp_step = 200.0 / leg_frames
        seq = []
        x = 100.0
        direction = 1
        for _leg in range(legs):
            for _ in range(leg_frames):
                x += amp_step * direction
                seq.append(_box_at(x, 300.0))
            direction *= -1
        rows = _trk(1, seq)
        events = derive_irrational_events(
            rows, person_by_tracklet={}, fps=FPS, config=_cfg(erratic_tortuosity=1e9)
        )
        irr = [e for e in events if e.category == CATEGORY_IRRATIONELL]
        assert len(irr) >= 1
        assert "oscillation" in irr[0].evidence["sub_signals"]
        assert irr[0].evidence["sub_signals"]["oscillation"]["reversals"] >= 3

    def test_monotonic_path_does_not_fire_oscillation(self):
        n = 150
        seq = [_box_at(100.0 + i * 5.0, 300.0) for i in range(n)]
        rows = _trk(1, seq)
        events = derive_irrational_events(rows, person_by_tracklet={}, fps=FPS, config=_cfg())
        assert not any(
            "oscillation" in e.evidence["sub_signals"] for e in events if e.category == CATEGORY_IRRATIONELL
        )


class TestFreezeAndBolt:
    def test_stillness_then_bolt_fires_freeze_bolt(self):
        # 4s stationary (>= freeze_time_s=3s), then a sudden sprint.
        still = _stationary(40, 300.0, 300.0)
        bolt = [_box_at(300.0 + i * 40.0, 300.0) for i in range(1, 60)]  # 400 px/s = 5 bh/s
        rows = _trk(1, still + bolt)
        events = derive_irrational_events(
            rows, person_by_tracklet={}, fps=FPS, config=_cfg(erratic_tortuosity=1e9)
        )
        irr = [e for e in events if e.category == CATEGORY_IRRATIONELL]
        assert len(irr) >= 1
        assert any("freeze_bolt" in e.evidence["sub_signals"] for e in irr)

    def test_stillness_without_bolt_does_not_fire_freeze_bolt(self):
        # Stationary throughout, never bolts — STILLA territory, not this
        # sub-signal (and STILLA precedence would suppress it anyway if fed).
        rows = _trk(1, _stationary(100, 300.0, 300.0))
        events = derive_irrational_events(rows, person_by_tracklet={}, fps=FPS, config=_cfg())
        assert not any(
            "freeze_bolt" in e.evidence["sub_signals"] for e in events if e.category == CATEGORY_IRRATIONELL
        )

    def test_brief_pause_below_freeze_time_does_not_fire(self):
        # Only 1s stationary (below freeze_time_s=3s) before bolting.
        still = _stationary(10, 300.0, 300.0)
        bolt = [_box_at(300.0 + i * 40.0, 300.0) for i in range(1, 60)]
        rows = _trk(1, still + bolt)
        events = derive_irrational_events(rows, person_by_tracklet={}, fps=FPS, config=_cfg())
        assert not any(
            "freeze_bolt" in e.evidence["sub_signals"] for e in events if e.category == CATEGORY_IRRATIONELL
        )


class TestEnsembleAndPrecedence:
    def test_evidence_never_a_bare_label(self):
        n = 150
        radius = 15.0
        step = 1.5
        seq = [
            _box_at(300.0 + radius * math.cos(i * step), 300.0 + radius * math.sin(i * step))
            for i in range(n)
        ]
        rows = _trk(1, seq)
        events = derive_irrational_events(rows, person_by_tracklet={}, fps=FPS, config=_cfg())
        irr = [e for e in events if e.category == CATEGORY_IRRATIONELL]
        assert irr
        for e in irr:
            assert e.evidence["sub_signals"], "evidence must name which sub-signal(s) fired"
            assert e.evidence["summary"] != "irrational behavior (unspecified)"

    def test_confidence_in_zero_to_one(self):
        n = 150
        seq = [_box_at(300.0 + 15.0 * math.cos(i * 1.5), 300.0 + 15.0 * math.sin(i * 1.5)) for i in range(n)]
        rows = _trk(1, seq)
        events = derive_irrational_events(rows, person_by_tracklet={}, fps=FPS, config=_cfg())
        for e in events:
            assert 0.0 <= e.confidence <= 1.0

    def test_person_id_tagged_when_p3_ran(self):
        n = 150
        seq = [_box_at(300.0 + 15.0 * math.cos(i * 1.5), 300.0 + 15.0 * math.sin(i * 1.5)) for i in range(n)]
        rows = _trk(7, seq)
        events = derive_irrational_events(rows, person_by_tracklet={7: 3}, fps=FPS, config=_cfg())
        irr = [e for e in events if e.category == CATEGORY_IRRATIONELL]
        assert irr and all(e.person_id == 3 for e in irr)

    def test_stilla_suppresses_irrational_on_same_frames(self):
        # A trajectory that would otherwise fire erratic/oscillation (small
        # circling) is entirely covered by a STILLA span for every one of
        # its frames — report §4's precedence rule means no IRRATIONELL
        # event should survive.
        n = 150
        seq = [_box_at(300.0 + 15.0 * math.cos(i * 1.5), 300.0 + 15.0 * math.sin(i * 1.5)) for i in range(n)]
        rows = _trk(1, seq)
        all_frames = {int(r["frame_no"]) for r in rows}
        events = derive_irrational_events(
            rows,
            person_by_tracklet={},
            fps=FPS,
            config=_cfg(),
            still_frames_by_tracklet={1: all_frames},
        )
        assert not any(e.category == CATEGORY_IRRATIONELL for e in events)

    def test_stilla_does_not_suppress_other_tracklets(self):
        # Precedence is per-tracklet, per-frame — suppressing tracklet 1's
        # frames must not touch tracklet 2's independent erratic signal.
        n = 150
        seq1 = [_box_at(300.0 + 15.0 * math.cos(i * 1.5), 300.0 + 15.0 * math.sin(i * 1.5)) for i in range(n)]
        seq2 = [_box_at(700.0 + 15.0 * math.cos(i * 1.5), 700.0 + 15.0 * math.sin(i * 1.5)) for i in range(n)]
        rows = _trk(1, seq1) + _trk(2, seq2)
        all_frames_t1 = {int(r["frame_no"]) for r in rows if r["tracklet_id"] == 1}
        events = derive_irrational_events(
            rows,
            person_by_tracklet={},
            fps=FPS,
            config=_cfg(),
            still_frames_by_tracklet={1: all_frames_t1},
        )
        irr = [e for e in events if e.category == CATEGORY_IRRATIONELL]
        assert irr and all(e.evidence["tracklet_id"] == 2 for e in irr)


class TestDeterminism:
    def test_same_input_twice_byte_identical(self):
        import json

        n = 150
        subject = [_box_at(500.0 - i * 5.0, 300.0) for i in range(n)]
        member_a = [_box_at(300.0 + i * 5.0, 260.0) for i in range(n)]
        member_b = [_box_at(300.0 + i * 5.0, 300.0) for i in range(n)]
        member_c = [_box_at(300.0 + i * 5.0, 340.0) for i in range(n)]
        rows = _trk(1, subject) + _trk(2, member_a) + _trk(3, member_b) + _trk(4, member_c)

        def run():
            events = derive_irrational_events(
                rows, person_by_tracklet={1: 10, 2: 11, 3: 12, 4: 13}, fps=FPS, config=_cfg()
            )
            return json.dumps([e.to_dict() for e in events], sort_keys=True)

        assert run() == run()
