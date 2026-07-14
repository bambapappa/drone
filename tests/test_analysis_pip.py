"""Analysis package: PiP detection.

Carried forward from tests/test_pip.py — same logic, now importing from analysis.
"""

import cv2
import numpy as np

from analysis.pip import PipAutoDetector, detect_pip_frame, split_active_roi

H, W = 540, 960


def colorful(rng):
    """A saturated (colour) aerial-like frame."""
    hsv = np.zeros((H, W, 3), np.uint8)
    hsv[..., 0] = rng.integers(0, 180, (H, W))
    hsv[..., 1] = rng.integers(120, 255, (H, W))  # high saturation
    hsv[..., 2] = rng.integers(80, 255, (H, W))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def add_gray_inset(frame, x0f, y0f, x1f, y1f, rng):
    """Paint a grayscale (IR-like) rectangle into the frame."""
    x0, y0, x1, y1 = int(x0f * W), int(y0f * H), int(x1f * W), int(y1f * H)
    g = rng.integers(40, 210, (y1 - y0, x1 - x0, 1), dtype=np.uint8)
    frame[y0:y1, x0:x1] = np.repeat(g, 3, axis=2)  # R=G=B => zero saturation
    return frame


def test_detects_top_right_corner():
    rng = np.random.default_rng(0)
    det = PipAutoDetector()
    for _ in range(12):
        det.feed(add_gray_inset(colorful(rng), 0.66, 0.0, 1.0, 0.45, rng))
    assert det.locked and det.region is not None
    assert det.layout == "top-right"
    x, y, w, h = det.region
    assert x > 0.5 and y < 0.1  # anchored top-right


def test_detects_split_right():
    rng = np.random.default_rng(1)
    det = PipAutoDetector()
    for _ in range(12):
        det.feed(add_gray_inset(colorful(rng), 0.5, 0.0, 1.0, 1.0, rng))
    assert det.layout == "split-right"
    assert det.region == (0.5, 0.0, 0.5, 1.0)


def test_no_false_positive_on_colorful():
    rng = np.random.default_rng(2)
    det = PipAutoDetector()
    for _ in range(20):
        det.feed(colorful(rng))
    assert not det.locked and det.region is None


def test_detects_pip_appearing_late():
    """The inset turns on partway through the feed (observed ~40 s in one
    film): colour-only first, then the gray corner appears — must still lock."""
    rng = np.random.default_rng(7)
    det = PipAutoDetector()
    for _ in range(15):  # no inset yet
        det.feed(colorful(rng))
    assert not det.locked
    for _ in range(12):  # inset now present
        det.feed(add_gray_inset(colorful(rng), 0.66, 0.0, 1.0, 0.45, rng))
    assert det.locked and det.layout == "top-right"


def test_transient_not_locked():
    """A single odd frame among colourful ones must not lock a region."""
    rng = np.random.default_rng(4)
    det = PipAutoDetector(window=10, need=4)
    for i in range(20):
        if i == 5:
            det.feed(add_gray_inset(colorful(rng), 0.66, 0, 1.0, 0.45, rng))
        else:
            det.feed(colorful(rng))
    assert not det.locked  # one stray positive never reaches need=4


def test_split_active_roi_mapping():
    assert split_active_roi("split-right") == (0.0, 0.0, 0.5, 1.0)  # IR right -> keep left
    assert split_active_roi("split-left") == (0.5, 0.0, 0.5, 1.0)  # IR left -> keep right
    assert split_active_roi("top-right") is None  # corner -> not a crop
    assert split_active_roi("") is None


def test_split_detection_feeds_active_crop():
    """End-to-end of the split path's pure logic."""
    rng = np.random.default_rng(11)
    det = PipAutoDetector()
    for _ in range(12):
        det.feed(add_gray_inset(colorful(rng), 0.5, 0.0, 1.0, 1.0, rng))
    assert det.layout == "split-right"
    assert split_active_roi(det.layout) == (0.0, 0.0, 0.5, 1.0)


def test_single_frame_helper():
    rng = np.random.default_rng(5)
    fr = add_gray_inset(colorful(rng), 0.0, 0.0, 0.34, 0.46, rng)
    res = detect_pip_frame(fr)
    assert res is not None and res[0] == "top-left"
