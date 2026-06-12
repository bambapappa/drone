"""One Euro filter and global motion estimation."""

import numpy as np

from app.vision.flow import BoxFilter, GlobalMotion, OneEuroFilter


def test_one_euro_smooths_jitter():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    rng = np.random.default_rng(0)
    out = [f(100 + rng.normal(0, 2), i * 0.04) for i in range(100)]
    # Output variance well below input noise variance
    assert np.std(out[20:]) < 1.5


def test_one_euro_follows_fast_motion():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    x = 0.0
    for i in range(50):
        x = f(i * 10.0, i * 0.04)  # 250 px/s
    assert abs(x - 49 * 10.0) < 25  # small lag only


def test_box_filter_step_glides():
    """A detection correction (step input) is absorbed over a few frames,
    not teleported in one."""
    bf = BoxFilter()
    t = 0.0
    for i in range(10):
        bf((100, 100, 50, 80), t)
        t += 1 / 24
    first = bf((140, 100, 50, 80), t)[0]
    assert first < 120  # first frame covers only part of the step
    for i in range(12):
        t += 1 / 24
        last = bf((140, 100, 50, 80), t)
    assert abs(last[0] - 140) < 3  # converged within ~0.5 s


def test_box_filter_huge_step_is_slew_limited():
    """Even a giant correction never teleports: per-frame motion is capped
    relative to box size."""
    bf = BoxFilter(slew=3.0)
    t = 0.0
    for i in range(5):
        bf((100, 100, 50, 80), t)
        t += 1 / 24
    nxt = bf((600, 100, 50, 80), t)[0]
    assert nxt - 100 <= 3.0 * 80 * (1 / 24) + 1e-6  # ≤ slew * dim * dt


def test_box_filter_feed_forward_no_lag():
    """Camera/scene motion fed forward passes through 1:1 — no trailing."""
    bf = BoxFilter()
    t, x = 0.0, 100.0
    bf((x, 100, 50, 80), t)
    out = x
    for i in range(24):
        x += 8.0  # scene shifts 8 px/frame
        t += 1 / 24
        out = bf((x, 100, 50, 80), t, ff=(8.0, 0.0))[0]
    assert abs(out - x) < 1.0


def test_box_filter_smooth_pursuit_low_lag():
    """Constant motion is followed with small lag."""
    bf = BoxFilter()
    t, x = 0.0, 100.0
    out = 100.0
    for i in range(48):  # 2 s at 24 fps, 120 px/s
        x += 120 / 24
        t += 1 / 24
        out = bf((x, 100, 50, 80), t)[0]
    assert x - out < 15  # lag under ~0.12 s of motion


def test_box_filter_reset_to():
    bf = BoxFilter()
    bf((100, 100, 50, 80), 0.0)
    bf.reset_to((500, 500, 40, 70), 1.0)
    cx, cy, w, h = bf((505, 500, 40, 70), 1.05)
    assert 500 <= cx < 506


def test_global_motion_translation():
    gm = GlobalMotion()
    rng = np.random.default_rng(1)
    base = (rng.random((120, 160)) * 255).astype(np.uint8)
    import cv2

    base = cv2.GaussianBlur(base, (5, 5), 0)
    shifted = np.roll(base, 3, axis=1)  # camera pan: scene moves +3 px in x
    gm.update(base, scale=2.0)
    dx, dy = gm.update(shifted, scale=2.0)
    assert abs(dx - 6.0) < 1.5  # 3 px at half-res => 6 source px
    assert abs(dy) < 1.5
    # Stabilized coords: a point moving with the scene stays put
    sx0, _ = gm.to_stab(106.0, 50.0)
    assert abs(sx0 - 100.0) < 2.0
