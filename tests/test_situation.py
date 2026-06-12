"""Situation analyzer: fire/smoke heuristics and base suggestion."""

import numpy as np

from app.vision.situation import SituationAnalyzer, fire_mask


def solid(bgr, w=320, h=180):
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:] = bgr
    return f


def test_fire_mask_hits_fire_colors():
    frame = solid((20, 60, 220))  # strong red/orange (BGR)
    assert fire_mask(frame).mean() > 0.9
    frame = solid((200, 200, 200))  # gray
    assert fire_mask(frame).mean() < 0.05


def test_fire_reported_after_hold():
    an = SituationAnalyzer(min_area=0.004, hold_s=1.0)
    frame = solid((40, 40, 40))
    frame[40:90, 100:180] = (20, 60, 230)  # fire patch
    t = 0.0
    state = None
    for i in range(12):
        state = an.update(frame, t, None)
        t += 0.2
    assert state.fire is not None
    assert 0.2 < state.fire.pos[0] < 0.7


def test_no_fire_on_neutral_frame():
    an = SituationAnalyzer(hold_s=0.5)
    frame = solid((90, 90, 90))
    state = None
    for i in range(8):
        state = an.update(frame, i * 0.2, None)
    assert state.fire is None


def test_base_opposite_danger():
    an = SituationAnalyzer()
    frame = solid((90, 90, 90))
    state = None
    for i in range(5):
        state = an.update(frame, i * 0.2, (0.9, 0.5))  # danger right side
    assert state.base is not None
    assert state.base[0] < 0.3  # suggestion on the left
    assert state.base_reasons


def test_no_base_without_information():
    an = SituationAnalyzer()
    frame = solid((90, 90, 90))
    state = None
    for i in range(5):
        state = an.update(frame, i * 0.2, None)
    assert state.base is None
