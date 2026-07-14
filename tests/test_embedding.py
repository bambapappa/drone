"""Tests for the per-detection appearance embedding (Phase 1 identity substrate).

Covers the floor/fallback decision (the part that genuinely needs a guard — a
wrong floor silently routes everything to the weak HSV space), the HSV
embedder (pure cv2/numpy, no torch), and make_embedder's selection logic. The
real OSNet path is not unit-tested here — it needs torch + weights, exactly as
P1's real YOLO Detector isn't unit-tested (only FakeDetector is). OSNet is
exercised in a real run with weights present; the orchestrator tests drive P3
through a FakeEmbedder.
"""

from __future__ import annotations

import numpy as np

from analysis.embedding import (
    HSVEmbedder,
    OSNetEmbedder,
    below_reid_floor,
    make_embedder,
)


def _frame_with_box(box=(10, 10, 60, 90), size=(120, 120, 3)) -> np.ndarray:
    """A synthetic frame with a colored rectangle so HSV has signal."""
    frame = np.full(size, 60, dtype=np.uint8)
    x0, y0, x1, y1 = box
    frame[y0:y1, x0:x1] = (0, 0, 200)  # red rectangle (BGR)
    return frame


class TestReidFloor:
    def test_small_crop_below_floor(self):
        assert below_reid_floor((0, 0, 20, 25), floor=32) is True

    def test_large_crop_above_floor(self):
        assert below_reid_floor((0, 0, 60, 90), floor=32) is False

    def test_boundary_equal_not_below(self):
        # exactly 32 on both sides is NOT below the floor (min side == floor)
        assert below_reid_floor((0, 0, 32, 40), floor=32) is False


class TestHSVEmbedder:
    def test_produces_normalized_vector_with_method_tag(self):
        emb = HSVEmbedder()
        frame = _frame_with_box()
        result = emb.embed(frame, (10, 10, 60, 90))
        assert result is not None
        assert result.method == "hsv"
        # L2-normalized
        assert abs(np.linalg.norm(result.vector) - 1.0) < 1e-6

    def test_degenerate_crop_returns_none(self):
        emb = HSVEmbedder()
        frame = _frame_with_box()
        # far too small for even the HSV fallback (appearance_hist floor ~4x8)
        assert emb.embed(frame, (0, 0, 2, 2)) is None


class TestMakeEmbedder:
    def test_no_weights_returns_hsv_embedder(self, tmp_path):
        from analysis.orchestrator import OfflineConfig

        cfg = OfflineConfig(reid_weights=None)
        emb = make_embedder(cfg)
        assert isinstance(emb, HSVEmbedder)

    def test_missing_weights_file_falls_back_to_hsv(self, tmp_path):
        # A configured-but-absent weights path must degrade to HSV (not crash),
        # so a run whose manifest can't back an OSNet claim uses HSV honestly.
        from analysis.orchestrator import OfflineConfig

        cfg = OfflineConfig(reid_weights=str(tmp_path / "nonexistent.pth"))
        emb = make_embedder(cfg)
        assert isinstance(emb, HSVEmbedder)

    def test_present_weights_returns_osnet_embedder(self, tmp_path):
        from analysis.orchestrator import OfflineConfig

        weights = tmp_path / "osnet.pth"
        weights.write_bytes(b"not-a-real-model")  # existence check only
        cfg = OfflineConfig(reid_weights=str(weights), reid_floor=32)
        emb = make_embedder(cfg)
        assert isinstance(emb, OSNetEmbedder)
        assert emb.floor == 32


class TestOSNetEmbedderFallbackRouting:
    """The floor routing is testable without torch: a small crop must route to
    HSV, bypassing the (unloaded) OSNet model entirely."""

    def test_small_crop_uses_hsv_without_loading_model(self):
        # Construct without ever loading torch; embed() on a small crop must
        # hit the HSV path before _ensure_loaded() is ever called.
        emb = OSNetEmbedder(weights_path="/nonexistent.pt", floor=32)
        frame = _frame_with_box()
        result = emb.embed(frame, (10, 10, 20, 25))  # 10x15 < floor 32
        assert result is not None
        assert result.method == "hsv"
        assert emb._model is None  # OSNet model never loaded
