"""Person registry: stable IDs and appearance-based re-identification."""

import numpy as np

from app.vision.registry import PersonRegistry

DIAG = 1000.0


def hist_for(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = rng.random(128).astype(np.float32)
    return h / np.linalg.norm(h)


def test_same_track_same_person():
    reg = PersonRegistry()
    h = hist_for(1)
    reg.begin_frame()
    p1 = reg.resolve(10, 0.0, h, (100, 100), DIAG)
    reg.begin_frame()
    p2 = reg.resolve(10, 0.5, h, (110, 100), DIAG)
    assert p1 == p2
    assert reg.unique_total == 1


def test_two_tracks_two_persons():
    reg = PersonRegistry()
    reg.begin_frame()
    p1 = reg.resolve(10, 0.0, hist_for(1), (100, 100), DIAG)
    p2 = reg.resolve(11, 0.0, hist_for(2), (500, 500), DIAG)
    assert p1 != p2
    assert reg.unique_total == 2


def test_reentry_same_appearance_reidentified():
    reg = PersonRegistry()
    h = hist_for(1)
    reg.begin_frame()
    p1 = reg.resolve(10, 0.0, h, (100, 100), DIAG)
    # Track 10 dies; 3 s later a new track with same appearance appears nearby.
    reg.begin_frame()
    p2 = reg.resolve(20, 3.0, h, (150, 120), DIAG)
    assert p1 == p2
    assert reg.unique_total == 1


def test_reentry_different_appearance_new_person():
    reg = PersonRegistry()
    reg.begin_frame()
    reg.resolve(10, 0.0, hist_for(1), (100, 100), DIAG)
    reg.begin_frame()
    p2 = reg.resolve(20, 3.0, hist_for(99), (150, 120), DIAG)
    assert p2 == 2
    assert reg.unique_total == 2


def test_no_reid_while_track_active():
    """An identical-looking person while the first is still tracked => two people."""
    reg = PersonRegistry()
    h = hist_for(1)
    reg.begin_frame()
    p1 = reg.resolve(10, 0.0, h, (100, 100), DIAG)
    p2 = reg.resolve(11, 0.0, h.copy(), (800, 800), DIAG)
    assert p1 != p2


def test_reentry_too_far_away_is_new_person():
    reg = PersonRegistry()
    h = hist_for(1)
    reg.begin_frame()
    reg.resolve(10, 0.0, h, (0, 0), DIAG)
    reg.begin_frame()
    # Same look but impossibly far for the elapsed time (0.5 s, ~9000 px)
    p2 = reg.resolve(20, 0.5, h.copy(), (9000, 0), DIAG)
    assert p2 == 2
