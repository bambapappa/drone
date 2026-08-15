"""Behavior tests for the local scene-coordinate contract (DECISIONS B29)."""

from __future__ import annotations

import cv2
import numpy as np

from analysis.scene import SceneGMC, SceneMotionAccumulator, scene_measurement, transform_point
from analysis.tracker import Tracker


def test_accumulates_camera_motion_and_round_trips_points():
    acc = SceneMotionAccumulator()
    first = acc.advance(frame_no=0, pairwise=np.eye(2, 3), confidence=1.0)
    assert first.segment == 0
    assert first.linked
    assert first.link_method == "initial"

    # Static scenery moves +12/-4 pixels on screen between the two frames.
    warp = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -4.0]])
    second = acc.advance(frame_no=1, pairwise=warp, confidence=0.8)
    assert second.segment == 0
    assert second.linked
    assert second.confidence == 0.8
    assert second.link_method == "sparse_flow"

    frame_xy = transform_point(second.scene_to_frame, (100.0, 50.0))
    assert frame_xy == (112.0, 46.0)
    assert transform_point(second.frame_to_scene, frame_xy) == (100.0, 50.0)


def test_failed_link_starts_new_segment_instead_of_guessing():
    acc = SceneMotionAccumulator()
    acc.advance(frame_no=0, pairwise=np.eye(2, 3), confidence=1.0)
    acc.advance(
        frame_no=1,
        pairwise=np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0]]),
        confidence=0.9,
    )
    lost = acc.advance(frame_no=2, pairwise=None, confidence=0.0)

    assert lost.segment == 1
    assert not lost.linked
    assert lost.link_method == "unlinked"
    assert np.allclose(lost.scene_to_frame, np.eye(3))
    assert np.allclose(lost.frame_to_scene, np.eye(3))


def test_scene_measurement_transforms_foot_and_body_height():
    # Frame is translated +10 in x relative to its local scene.
    frame_to_scene = np.array([[1.0, 0.0, -10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    measurement = scene_measurement([100.0, 20.0, 140.0, 100.0], frame_to_scene, segment=3)

    assert measurement["scene_pos"] == [110.0, 100.0]
    assert measurement["scene_box_h"] == 80.0
    assert measurement["scene_segment"] == 3


def test_scene_gmc_recovers_translation_from_real_image_flow():
    rng = np.random.default_rng(7)
    first = rng.integers(0, 256, size=(180, 240), dtype=np.uint8)
    shift = np.array([[1.0, 0.0, 9.0], [0.0, 1.0, -5.0]])
    second = cv2.warpAffine(first, shift, (240, 180))

    gmc = SceneGMC(downscale=1.0, seed=7)
    gmc.apply(first)
    estimated = gmc.apply(second)

    assert gmc.current is not None
    assert gmc.current.segment == 0
    assert gmc.current.linked
    assert gmc.current.confidence > 0.5
    assert gmc.current.link_method == "sparse_flow"
    assert np.allclose(estimated[:, 2], [9.0, -5.0], atol=0.5)
    # A static scene point which moved on screen maps back to its first-frame
    # coordinate through the persisted inverse transform.
    assert np.allclose(transform_point(gmc.current.frame_to_scene, (109.0, 75.0)), (100.0, 80.0), atol=0.5)


def test_scene_gmc_recovers_small_rotation():
    rng = np.random.default_rng(11)
    first = rng.integers(0, 256, size=(220, 260), dtype=np.uint8)
    expected = cv2.getRotationMatrix2D((130.0, 110.0), 3.0, 1.0)
    second = cv2.warpAffine(first, expected, (260, 220))
    gmc = SceneGMC(downscale=1.0, seed=11)

    gmc.apply(first)
    estimated = gmc.apply(second)

    assert gmc.current is not None
    assert gmc.current.segment == 0
    assert gmc.current.confidence > 0.5
    assert gmc.current.link_method == "sparse_flow"
    assert np.allclose(estimated, expected, atol=0.5)


def test_scene_gmc_starts_new_segment_after_textureless_frame():
    blank = np.zeros((120, 160), dtype=np.uint8)
    textured = np.random.default_rng(3).integers(0, 256, size=blank.shape, dtype=np.uint8)
    gmc = SceneGMC(downscale=1.0)

    gmc.apply(blank)
    gmc.apply(textured)

    assert gmc.current is not None
    assert gmc.current.segment == 1
    assert not gmc.current.linked
    assert gmc.current.confidence == 0.0


def test_scene_gmc_uses_strong_feature_fallback_after_flow_failure(monkeypatch):
    rng = np.random.default_rng(31)
    first = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
    expected = cv2.getRotationMatrix2D((160.0, 120.0), 2.0, 1.0)
    expected[:, 2] += [13.0, -7.0]
    second = cv2.warpAffine(first, expected, (320, 240))
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", lambda *args, **kwargs: (None, None, None))
    gmc = SceneGMC(downscale=1.0, seed=31)

    gmc.apply(first)
    estimated = gmc.apply(second)

    assert gmc.current is not None
    assert gmc.current.segment == 0
    assert gmc.current.linked
    assert gmc.current.confidence > 0.5
    assert gmc.current.link_method == "sift_ransac"
    assert np.allclose(estimated, expected, atol=0.6)


def test_scene_gmc_feature_fallback_does_not_bridge_unrelated_frames(monkeypatch):
    first = np.random.default_rng(41).integers(0, 256, size=(240, 320), dtype=np.uint8)
    second = np.random.default_rng(42).integers(0, 256, size=(240, 320), dtype=np.uint8)
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", lambda *args, **kwargs: (None, None, None))
    gmc = SceneGMC(downscale=1.0, seed=41)

    gmc.apply(first)
    estimated = gmc.apply(second)

    assert gmc.current is not None
    assert gmc.current.segment == 1
    assert not gmc.current.linked
    assert gmc.current.confidence == 0.0
    assert gmc.current.link_method == "unlinked"
    assert np.array_equal(estimated, np.eye(2, 3))


def test_prepared_scene_warp_is_consumed_without_advancing_twice():
    blank = np.zeros((120, 160), dtype=np.uint8)
    textured = np.random.default_rng(3).integers(0, 256, size=blank.shape, dtype=np.uint8)
    gmc = SceneGMC(downscale=1.0)

    gmc.prepare(blank)
    gmc.apply(blank)
    gmc.prepare(textured)
    consumed = gmc.apply(textured)

    assert gmc.current is not None
    assert gmc.current.frame_no == 1
    assert gmc.current.segment == 1
    assert not gmc.current.linked
    assert np.array_equal(consumed, np.eye(2, 3))


def test_segment_reset_clears_associations_without_reusing_tracker_session():
    class FakeBotSort:
        def __init__(self):
            self.tracked_stracks = [object()]
            self.lost_stracks = [object()]
            self.removed_stracks = [object()]
            self.frame_id = 7
            self.kalman_filter = object()

        def get_kalmanfilter(self):
            return "fresh"

    tracker = Tracker.__new__(Tracker)
    tracker._tracker = FakeBotSort()
    tracker._reset_association_state()

    assert tracker._tracker.tracked_stracks == []
    assert tracker._tracker.lost_stracks == []
    assert tracker._tracker.removed_stracks == []
    assert tracker._tracker.kalman_filter == "fresh"
    assert tracker._tracker.frame_id == 7
