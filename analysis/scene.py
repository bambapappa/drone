"""Local scene coordinates derived from P2 camera motion (DECISIONS B29).

The scene space is deliberately *local*, not geographic.  It gives every
visually connected stretch of video a stable coordinate system whose first
frame is the origin.  If optical-flow GMC cannot link two adjacent frames we
start a new segment instead of accumulating a guessed transform.

`SceneGMC` implements the small ``apply(frame, detections) -> 2x3`` interface
that Ultralytics BoT-SORT expects.  This makes the transform used to compensate
the tracker the same transform persisted for review/heatmap/behavior; there is
one camera-motion estimate, not two implementations that can drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


def _matrix3(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape == (2, 3):
        arr = np.vstack([arr, [0.0, 0.0, 1.0]])
    if arr.shape != (3, 3):
        raise ValueError("scene transform must be 2x3 or 3x3")
    if not np.isfinite(arr).all():
        raise ValueError("scene transform contains non-finite values")
    scale = arr[2, 2]
    if abs(scale) < 1e-12:
        raise ValueError("scene transform is singular")
    return arr / scale


def transform_point(
    matrix: np.ndarray | list[list[float]], point: tuple[float, float]
) -> tuple[float, float]:
    """Project one 2D point through an affine/homogeneous transform."""

    mat = _matrix3(np.asarray(matrix, dtype=np.float64))
    out = mat @ np.array([float(point[0]), float(point[1]), 1.0])
    if abs(out[2]) < 1e-12:
        raise ValueError("point projects to infinity")
    return float(out[0] / out[2]), float(out[1] / out[2])


@dataclass(frozen=True)
class SceneFrame:
    frame_no: int
    segment: int
    scene_to_frame: np.ndarray
    frame_to_scene: np.ndarray
    confidence: float
    linked: bool

    def to_record(self, pts_ms: float | None = None) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "frame_no": self.frame_no,
            "scene_segment": self.segment,
            "scene_to_frame": [[round(float(v), 8) for v in row] for row in self.scene_to_frame],
            "frame_to_scene": [[round(float(v), 8) for v in row] for row in self.frame_to_scene],
            "scene_confidence": round(float(self.confidence), 4),
            "scene_linked": self.linked,
        }
        if pts_ms is not None:
            rec["pts_ms"] = round(float(pts_ms), 3)
        return rec


class SceneMotionAccumulator:
    """Accumulate pairwise camera warps into honest local scene segments."""

    def __init__(self) -> None:
        self._last_frame = -1
        self._segment = 0
        self._scene_to_frame = np.eye(3, dtype=np.float64)

    def advance(
        self,
        frame_no: int,
        pairwise: np.ndarray | None,
        confidence: float,
    ) -> SceneFrame:
        if frame_no != self._last_frame + 1:
            raise ValueError("scene frames must be accumulated sequentially")

        linked = True
        if self._last_frame < 0:
            self._scene_to_frame = np.eye(3, dtype=np.float64)
        elif pairwise is None:
            # B29: never bridge a visual loss by pretending identity motion.
            self._segment += 1
            self._scene_to_frame = np.eye(3, dtype=np.float64)
            linked = False
            confidence = 0.0
        else:
            self._scene_to_frame = _matrix3(pairwise) @ self._scene_to_frame
            self._scene_to_frame = _matrix3(self._scene_to_frame)

        try:
            frame_to_scene = np.linalg.inv(self._scene_to_frame)
        except np.linalg.LinAlgError as exc:
            raise ValueError("scene transform is not invertible") from exc

        self._last_frame = frame_no
        return SceneFrame(
            frame_no=frame_no,
            segment=self._segment,
            scene_to_frame=self._scene_to_frame.copy(),
            frame_to_scene=frame_to_scene,
            confidence=max(0.0, min(1.0, float(confidence))),
            linked=linked,
        )


def scene_measurement(
    xyxy: list[float] | tuple[float, float, float, float],
    frame_to_scene: np.ndarray | list[list[float]],
    segment: int,
) -> dict[str, Any]:
    """Convert a tracked box's foot point and body height to scene space."""

    x0, y0, x1, y1 = (float(v) for v in xyxy)
    foot = transform_point(frame_to_scene, ((x0 + x1) / 2.0, y1))
    top = transform_point(frame_to_scene, ((x0 + x1) / 2.0, y0))
    body_h = max(float(np.hypot(foot[0] - top[0], foot[1] - top[1])), 1.0)
    return {
        "scene_pos": [round(foot[0], 4), round(foot[1], 4)],
        "scene_box_h": round(body_h, 4),
        "scene_segment": int(segment),
    }


class SceneGMC:
    """BoT-SORT-compatible sparse-flow GMC with persisted scene provenance.

    The returned affine warp maps previous-frame pixels to current-frame
    pixels.  Quality is based on RANSAC inlier share and median reprojection
    error.  A rejected estimate returns identity to the tracker and starts a
    new scene segment for downstream consumers.
    """

    def __init__(self, downscale: float = 2.0, seed: int = 42) -> None:
        self.downscale = max(float(downscale), 1.0)
        self.seed = int(seed)
        self.accumulator = SceneMotionAccumulator()
        self.current: SceneFrame | None = None
        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None
        self._frame_no = -1
        self._feature_params = {
            "maxCorners": 1000,
            "qualityLevel": 0.01,
            "minDistance": 8,
            "blockSize": 3,
            "useHarrisDetector": False,
            "k": 0.04,
        }
        self._lk_params = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }

    def apply(self, raw_frame: np.ndarray, detections: list | None = None) -> np.ndarray:
        del detections  # Interface compatibility; sparse flow uses scene texture.
        self._frame_no += 1
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if raw_frame.ndim == 3 else raw_frame
        if self.downscale > 1.0:
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (max(1, int(w / self.downscale)), max(1, int(h / self.downscale))))
        points = cv2.goodFeaturesToTrack(gray, mask=None, **self._feature_params)

        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            self._prev_points = points
            identity = np.eye(2, 3, dtype=np.float64)
            self.current = self.accumulator.advance(self._frame_no, identity, confidence=1.0)
            return identity

        if self._prev_points is None:
            # The preceding frame had no usable texture.  There is no
            # correspondence from it to this frame, even if this frame has
            # good corners again, so B29 requires a new local segment.
            self._prev_gray = gray.copy()
            self._prev_points = points
            identity = np.eye(2, 3, dtype=np.float64)
            self.current = self.accumulator.advance(self._frame_no, None, confidence=0.0)
            return identity

        warp: np.ndarray | None = None
        confidence = 0.0
        try:
            matched, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_points, None, **self._lk_params
            )
            if matched is not None and status is not None:
                good = status.reshape(-1) == 1
                previous = self._prev_points.reshape(-1, 2)[good]
                current = matched.reshape(-1, 2)[good]
                finite = np.isfinite(previous).all(axis=1) & np.isfinite(current).all(axis=1)
                previous, current = previous[finite], current[finite]
                if len(previous) >= 8:
                    cv2.setRNGSeed(self.seed + self._frame_no)
                    estimated, inliers = cv2.estimateAffinePartial2D(
                        previous,
                        current,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                        maxIters=2000,
                        confidence=0.99,
                        refineIters=10,
                    )
                    if estimated is not None and inliers is not None:
                        inlier_mask = inliers.reshape(-1).astype(bool)
                        n_inliers = int(inlier_mask.sum())
                        ratio = n_inliers / len(previous)
                        if n_inliers >= 8 and ratio >= 0.35 and self._sane(estimated, gray.shape):
                            pred = previous @ estimated[:, :2].T + estimated[:, 2]
                            errors = np.linalg.norm(pred - current, axis=1)
                            median_error = float(np.median(errors[inlier_mask]))
                            confidence = ratio * max(0.0, 1.0 - median_error / 6.0)
                            warp = estimated.astype(np.float64)
                            warp[:, 2] *= self.downscale
        except cv2.error:
            warp = None
            confidence = 0.0

        self._prev_gray = gray.copy()
        self._prev_points = points
        self.current = self.accumulator.advance(self._frame_no, warp, confidence=confidence)
        return warp if warp is not None else np.eye(2, 3, dtype=np.float64)

    def _sane(self, warp: np.ndarray, shape: tuple[int, ...]) -> bool:
        if warp.shape != (2, 3) or not np.isfinite(warp).all():
            return False
        linear = warp[:, :2]
        det = float(np.linalg.det(linear))
        if not 0.25 <= abs(det) <= 4.0:
            return False
        h, w = shape[:2]
        return abs(float(warp[0, 2])) <= w * 0.75 and abs(float(warp[1, 2])) <= h * 0.75
