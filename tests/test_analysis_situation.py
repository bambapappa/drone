"""Analysis package: situation analyzer (fire/smoke heuristics and base suggestion).

Carried forward from tests/test_situation.py — same logic, now importing from analysis.
"""

import cv2
import numpy as np

from analysis.situation import SituationAnalyzer, fire_mask


def solid(bgr, w=320, h=180):
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:] = bgr
    return f


def test_fire_mask_hits_fire_colors():
    frame = solid((20, 60, 220))  # strong red/orange (BGR)
    assert fire_mask(frame).mean() > 0.9
    frame = solid((200, 200, 200))  # gray
    assert fire_mask(frame).mean() < 0.05


def fire_smoke_frame(rng, w=320, h=180):
    """Dark scene with a red fire patch and a turbulent gray smoke plume above it."""
    f = solid((40, 40, 40), w, h)
    f[120:150, 150:200] = (20, 60, 230)  # fire patch (BGR red/orange)
    blocks = rng.integers(110, 190, size=(14, 11, 1), dtype=np.uint8)
    plume = cv2.resize(blocks, (55, 70), interpolation=cv2.INTER_NEAREST)
    f[40:110, 150:205] = np.repeat(plume[:, :, None], 3, axis=2)
    return f


def test_fire_with_smoke_reported_after_hold():
    an = SituationAnalyzer(min_area=0.004, hold_s=1.0)
    rng = np.random.default_rng(0)
    t, state = 0.0, None
    for _ in range(12):
        state = an.update(fire_smoke_frame(rng), t, None)
        t += 0.2
    assert state.fire is not None
    assert 0.2 < state.fire.pos[0] < 0.8


def test_red_roof_without_smoke_not_fire():
    """A saturated red region with no smoke nearby (red tile roof) is rejected."""
    an = SituationAnalyzer(min_area=0.004, hold_s=1.0)
    frame = solid((40, 40, 40))
    frame[40:90, 100:180] = (20, 60, 230)  # red roof, no smoke anywhere
    state = None
    for i in range(12):
        state = an.update(frame, i * 0.2, None)
    assert state.fire is None


def test_fire_require_smoke_can_be_disabled():
    an = SituationAnalyzer(min_area=0.004, hold_s=1.0, fire_require_smoke=False)
    frame = solid((40, 40, 40))
    frame[40:90, 100:180] = (20, 60, 230)
    state = None
    for i in range(12):
        state = an.update(frame, i * 0.2, None)
    assert state.fire is not None


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


def _veg_frame(w=320, h=180):
    f = solid((40, 140, 40), w, h)  # green vegetation everywhere (BGR)
    return f


def test_base_prefers_through_road_with_exit():
    """A gray road spanning to both side edges should be chosen as base."""
    an = SituationAnalyzer()
    frame = _veg_frame()
    frame[80:110, :] = (120, 120, 120)  # horizontal road, full width => exits L/R
    state = None
    for i in range(4):
        state = an.update(frame, i * 0.2, (0.5, 0.05))  # danger at top
    assert state.base is not None
    assert 0.35 < state.base[1] < 0.75
    assert any("möjlig utväg" in r for r in state.base_reasons)


def test_base_warns_on_dead_end_pocket():
    """A small open pocket walled in by vegetation has no corridor out."""
    an = SituationAnalyzer()
    frame = _veg_frame()
    frame[70:95, 150:175] = (120, 120, 120)  # tiny open pocket, reaches no edge
    state = None
    for i in range(4):
        state = an.update(frame, i * 0.2, (0.1, 0.05))
    assert state.base is not None
    assert not any("möjlig utväg" in r for r in state.base_reasons)
