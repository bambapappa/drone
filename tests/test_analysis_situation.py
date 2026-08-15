"""Analysis package: situation analyzer (fire/smoke heuristics and base suggestion).

Carried forward from tests/test_situation.py — same logic, now importing from analysis.
"""

import cv2
import numpy as np

from analysis.situation import SituationAnalyzer, fire_mask, smoke_mask


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


def test_multiple_fire_blobs_are_exposed_as_separate_brand_signals():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    frame[40:80, 30:80] = (20, 60, 230)
    frame[100:150, 240:300] = (20, 60, 230)
    analyzer = SituationAnalyzer(min_area=0.004, hold_s=0.0, fire_require_smoke=False)

    state = analyzer.update(frame, 0.0, None)

    assert len(state.fire_hazards) == 2
    assert state.fire == state.fire_hazards[0]
    assert state.fire_hazards[0].pos != state.fire_hazards[1].pos


def test_hold_grace_is_not_reported_as_a_fresh_observation():
    fire = np.zeros((180, 320, 3), dtype=np.uint8)
    fire[40:100, 80:160] = (20, 60, 230)
    blank = np.zeros_like(fire)
    analyzer = SituationAnalyzer(min_area=0.004, hold_s=0.1, fire_require_smoke=False)

    analyzer.update(fire, 0.0, None)
    confirmed = analyzer.update(fire, 0.1, None)
    confirmed_observed = confirmed.fire.observed if confirmed.fire is not None else None
    held = analyzer.update(blank, 0.2, None)

    assert confirmed_observed is True
    assert held.fire is not None and held.fire.observed is False


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


def _frame_with_gray_block(shift=0, blob_shift=0, w=160, h=90):
    """Gray building-ish block + a gray blob that may drift in scene (WORK_W scale).

    `shift` is the camera pan: scene-static content moves +shift px in the
    image, matching warpAffine's forward-map convention for +dx affines."""
    img = np.full((h, w, 3), (40, 120, 60), np.uint8)  # greenish background
    cv2.rectangle(img, (10 + shift, 10), (60 + shift, 50), (90, 110, 105), -1)  # static gray building
    cv2.circle(img, (120 + blob_shift + shift, 60), 8, (100, 115, 110), -1)  # gray blob
    return img


def _shift_affine(dx, dy):
    return np.float32([[1, 0, dx], [0, 1, dy]])


class TestSceneCompensatedSmoke:
    def test_static_scene_camera_pan_no_smoke_when_compensated(self):
        # Camera pans +8px/frame; building and blob are scene-static.
        prev, cur = _frame_with_gray_block(0, 0), _frame_with_gray_block(8, 0)
        pg, g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        legacy = smoke_mask(prev, pg, g)  # uncompensated
        comp = smoke_mask(prev, pg, g, prev_to_cur=_shift_affine(8, 0))
        assert legacy.sum() > 0  # camera motion masquerades as motion — the measured failure
        assert comp.sum() == 0  # scene-static: no honest smoke motion

    def test_scene_drifting_blob_detected_when_compensated(self):
        # Camera pans +8; blob ALSO drifts +6 in scene (image move 14).
        prev, cur = _frame_with_gray_block(0, 0), _frame_with_gray_block(8, 6)
        pg, g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        comp = smoke_mask(prev, pg, g, prev_to_cur=_shift_affine(8, 0))
        assert comp.sum() > 0  # the 6px scene drift survives compensation

    def test_update_scene_motion_unlinked_pair_gives_empty_mask(self):
        sit = SituationAnalyzer(hold_s=0.0)
        s1 = sit.update(
            _frame_with_gray_block(0, 0), 0.0, danger_norm=None, prev_to_cur=None, scene_motion=True
        )
        # No honest motion signal this frame: smoke None — never guess across a
        # visual loss (B29 loss rule), hold() rides over single frames.
        assert s1.smoke is None


def _textured_gray_blob_frame(w=160, h=90, cx=60, cy=45, r=9, rng=None, smooth=False):
    """Green background with a gray blob at (cx, cy).

    `smooth=True`: flat interior (median |Laplacian| ~0). Otherwise: per-pixel
    noise inside the blob — turbulent internal structure, the way real smoke
    reads at WORK_W resolution (measured: fire-half blobs 16–68, traffic 3–7).
    """
    img = np.full((h, w, 3), (40, 120, 60), np.uint8)
    if smooth:
        cv2.circle(img, (cx, cy), r, (110, 115, 112), -1)
    else:
        if rng is None:
            rng = np.random.default_rng(0)
        noise = rng.integers(80, 190, size=(2 * r + 1, 2 * r + 1)).astype(np.uint8)
        region = img[cy - r : cy + r + 1, cx - r : cx + r + 1]
        mask = np.zeros(region.shape[:2], np.uint8)
        cv2.circle(mask, (r, r), r, 255, -1)
        gray3 = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        region[mask > 0] = gray3[mask > 0]
    return img


class TestWindowedReferenceAndTexture:
    """B32: 0.32 s windowed reference + in-blob texture gate (scene path only)."""

    def test_reference_is_window_back_not_previous_frame(self):
        # 10 fps, smoke_window_s=0.32 => the scene-path reference is ~3 frames
        # back, not the previous frame. A blob that makes a single-frame
        # excursion (moves between n-2 and n-1, back at n) is INVISIBLE over
        # the 3-frame span even though the consecutive diff would see it.
        ident = np.float32([[1, 0, 0], [0, 1, 0]])
        positions = [60] * 8 + [100, 60]  # single-frame excursion at n-1
        sit = SituationAnalyzer(hold_s=0.0, min_area=0.004, smoke_window_s=0.32, smoke_texture_min=0.0)
        state = None
        for i, cx in enumerate(positions):
            state = sit.update(
                _textured_gray_blob_frame(cx=cx),
                i * 0.1,
                None,
                prev_to_cur=ident,
                scene_motion=True,
                ref_lag=3,
            )
        assert state.smoke is None  # ref(0.6) and cur(0.9) both at 60: no span motion

    def test_blob_moved_across_window_is_detected(self):
        # Same setup, but the blob STAYS at its new position: over the 3-frame
        # span it has moved 40 px — detected (this is the measured reason k=8
        # works where k=1 fragments at ~1-2 px/frame drift).
        ident = np.float32([[1, 0, 0], [0, 1, 0]])
        positions = [60] * 8 + [100, 100]
        sit = SituationAnalyzer(hold_s=0.0, min_area=0.004, smoke_window_s=0.32, smoke_texture_min=0.0)
        state = None
        for i, cx in enumerate(positions):
            state = sit.update(
                _textured_gray_blob_frame(cx=cx),
                i * 0.1,
                None,
                prev_to_cur=ident,
                scene_motion=True,
                ref_lag=3,
            )
        assert state.smoke is not None

    def test_smooth_blob_rejected_by_texture_gate(self):
        # Same gray motion, flat interior: gray traffic/warp residue, not
        # smoke — the blob is gated exactly like fire_require_smoke gates fire.
        ident = np.float32([[1, 0, 0], [0, 1, 0]])
        sit = SituationAnalyzer(hold_s=0.0, min_area=0.004, smoke_texture_min=12.0)
        state = None
        for i in range(10):
            state = sit.update(
                _textured_gray_blob_frame(cx=60 + 4 * i, smooth=True),
                i * 0.1,
                None,
                prev_to_cur=ident,
                scene_motion=True,
                ref_lag=3,
            )
        assert state.smoke is None

    def test_textured_blob_passes_texture_gate(self):
        ident = np.float32([[1, 0, 0], [0, 1, 0]])
        rng = np.random.default_rng(7)
        sit = SituationAnalyzer(hold_s=0.0, min_area=0.004, smoke_texture_min=12.0)
        state = None
        for i in range(10):
            state = sit.update(
                _textured_gray_blob_frame(cx=60 + 4 * i, rng=rng),
                i * 0.1,
                None,
                prev_to_cur=ident,
                scene_motion=True,
                ref_lag=3,
            )
        assert state.smoke is not None

    def test_texture_gate_not_applied_on_legacy_path(self):
        # scene_motion=False must stay bit-identical to today: a smooth gray
        # moving blob IS legacy smoke (that behavior is pinned elsewhere).
        sit = SituationAnalyzer(hold_s=0.0, min_area=0.004, smoke_texture_min=12.0)
        state = None
        for i in range(10):
            state = sit.update(_textured_gray_blob_frame(cx=60 + 8 * i, smooth=True), i * 0.1, None)
        assert state.smoke is not None
