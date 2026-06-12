"""Behavior classification on synthetic tracks (no ML needed)."""

from app.vision.behavior import (
    STATUS_OK,
    STATUS_STILL,
    STATUS_TOWARD,
    BehaviorAnalyzer,
    BehaviorConfig,
)

CFG = BehaviorConfig(
    window_s=6.0,
    min_history_s=3.0,
    still_speed=0.10,
    still_time_s=4.0,
    toward_speed=0.25,
    toward_angle_deg=40.0,
    toward_time_s=1.5,
    prone_aspect=1.4,
)
DT = 0.2  # 5 Hz detection cadence
BOX_H = 50.0


def feed(an, pid, positions, danger=None, aspect=0.4, t0=0.0):
    status = prone = speed = None
    for i, (x, y) in enumerate(positions):
        status, prone, speed = an.update(pid, t0 + i * DT, (x, y), BOX_H, aspect, danger)
    return status, prone, speed


def test_still_person_flagged():
    an = BehaviorAnalyzer(CFG)
    # 8 s without motion (tiny jitter below threshold)
    pos = [(100 + (i % 2) * 0.3, 200) for i in range(40)]
    status, _, speed = feed(an, 1, pos)
    assert status == STATUS_STILL
    assert speed < 0.10


def test_walker_is_ok():
    an = BehaviorAnalyzer(CFG)
    # ~1 body-height/s to the right for 8 s
    pos = [(100 + i * DT * BOX_H, 200) for i in range(40)]
    status, _, speed = feed(an, 2, pos)
    assert status == STATUS_OK
    assert speed > 0.5


def test_short_history_not_flagged():
    an = BehaviorAnalyzer(CFG)
    pos = [(100, 200) for i in range(10)]  # only 2 s
    status, _, _ = feed(an, 3, pos)
    assert status == STATUS_OK


def test_moving_toward_danger_flagged():
    an = BehaviorAnalyzer(CFG)
    danger = (1000.0, 200.0)
    # walking straight at the danger point
    pos = [(100 + i * DT * BOX_H, 200) for i in range(40)]
    status, _, _ = feed(an, 4, pos, danger=danger)
    assert status == STATUS_TOWARD


def test_moving_away_from_danger_ok():
    an = BehaviorAnalyzer(CFG)
    danger = (0.0, 200.0)
    pos = [(100 + i * DT * BOX_H, 200) for i in range(40)]
    status, _, _ = feed(an, 5, pos, danger=danger)
    assert status == STATUS_OK


def test_still_wins_over_toward():
    """A still person near a danger point is STILL (not toward)."""
    an = BehaviorAnalyzer(CFG)
    danger = (110.0, 200.0)
    pos = [(100, 200) for _ in range(40)]
    status, _, _ = feed(an, 6, pos, danger=danger)
    assert status == STATUS_STILL


def test_prone_detected():
    an = BehaviorAnalyzer(CFG)
    pos = [(100, 200) for _ in range(40)]
    status, prone, _ = feed(an, 7, pos, aspect=2.0)
    assert status == STATUS_STILL
    assert prone


def test_drop_inactive():
    an = BehaviorAnalyzer(CFG)
    feed(an, 8, [(0, 0)] * 5)
    an.drop_inactive(set())
    assert an.status_of(8) == STATUS_OK
