"""Tests for the dwell/coverage heatmap computation (Phase 5, report §5.9)."""

from __future__ import annotations

from review.heatmap import compute_heatmap


def row(tid, frame_no, x0, y0, x1, y1):
    return {"tracklet_id": tid, "frame_no": frame_no, "xyxy": [x0, y0, x1, y1]}


def test_dwell_accumulates_per_cell():
    # 10 frames at 10 fps with the foot center in one cell → 1.0s dwell.
    rows = [row(1, f, 95.0, 100.0, 105.0, 200.0) for f in range(10)]
    hm = compute_heatmap(rows, fps=10.0, frame_w=1000, frame_h=500, grid_w=10)
    assert hm["grid_w"] == 10
    assert hm["grid_h"] == 5  # aspect-derived: 10 * 500/1000
    # foot = (100, 200) → cell (gx=1, gy=2) with 100px cells.
    assert hm["cell_s"][2][1] == 1.0
    assert hm["max_s"] == 1.0
    assert hm["total_s"] == 1.0


def test_person_filter_uses_mapping():
    rows = [row(1, 0, 0, 0, 10, 10), row(2, 0, 900, 0, 990, 480)]
    pmap = {1: 7, 2: 8}
    hm = compute_heatmap(
        rows, fps=1.0, frame_w=1000, frame_h=500, grid_w=10, person_id=7, person_by_tracklet=pmap
    )
    assert hm["total_s"] == 1.0
    assert hm["cell_s"][0][0] == 1.0  # only tracklet 1's cell
    hm_all = compute_heatmap(rows, fps=1.0, frame_w=1000, frame_h=500, grid_w=10)
    assert hm_all["total_s"] == 2.0


def test_out_of_frame_boxes_clamped():
    # Kalman-predicted boxes can extend past the frame — clamp, don't drop.
    rows = [row(1, 0, 980, 400, 1050, 520)]  # foot (1015, 520), past both edges
    hm = compute_heatmap(rows, fps=1.0, frame_w=1000, frame_h=500, grid_w=10)
    assert hm["total_s"] == 1.0
    assert hm["cell_s"][4][9] == 1.0  # last cell in both dimensions


def test_degenerate_inputs():
    assert compute_heatmap([], fps=0.0, frame_w=100, frame_h=100)["cell_s"] == []
    assert compute_heatmap([], fps=25.0, frame_w=0, frame_h=100)["cell_s"] == []
    hm = compute_heatmap([], fps=25.0, frame_w=100, frame_h=100)
    assert hm["total_s"] == 0.0 and hm["max_s"] == 0.0


def test_deterministic():
    rows = [
        row(t % 3, f, 10.0 * t, 5.0 * f, 10.0 * t + 20, 5.0 * f + 40) for t in range(3) for f in range(20)
    ]
    a = compute_heatmap(rows, fps=25.0, frame_w=640, frame_h=360)
    b = compute_heatmap(rows, fps=25.0, frame_w=640, frame_h=360)
    assert a == b
