"""Analysis package: tiling helpers.

Carried forward from tests/test_tiling.py — same logic, now importing from analysis.
"""

from analysis.tiling import nms_merge, tile_grid


def test_tile_grid_count_and_bounds():
    tiles = tile_grid(960, 540, 2)
    assert len(tiles) == 4
    for x0, y0, x1, y1 in tiles:
        assert 0 <= x0 < x1 <= 960
        assert 0 <= y0 < y1 <= 540


def test_tile_grid_covers_full_frame():
    # Union of tiles must reach all four edges (overlap means no gaps).
    tiles = tile_grid(1000, 800, 3)
    assert min(t[0] for t in tiles) == 0
    assert min(t[1] for t in tiles) == 0
    assert max(t[2] for t in tiles) == 1000
    assert max(t[3] for t in tiles) == 800


def test_tile_grid_overlap_present():
    # Adjacent tiles in a row should overlap horizontally.
    tiles = tile_grid(1000, 1000, 2)
    row = sorted(tiles, key=lambda t: (t[1], t[0]))[:2]
    assert row[0][2] > row[1][0]  # first tile's right edge past second's left


def test_nms_merge_dedups_overlapping():
    # Two near-identical boxes (same person from overlapping tiles) -> one.
    boxes = [[100, 100, 140, 200], [102, 101, 142, 201], [500, 300, 540, 400]]
    scores = [0.9, 0.7, 0.8]
    keep = nms_merge(boxes, scores, [0, 0, 0])
    assert len(keep) == 2
    assert 0 in keep  # higher-scoring of the duplicate pair survives
    assert 1 not in keep


def test_nms_merge_keeps_distinct():
    boxes = [[0, 0, 20, 40], [300, 300, 320, 340], [600, 100, 620, 140]]
    keep = nms_merge(boxes, [0.5, 0.6, 0.7], [0, 0, 0])
    assert len(keep) == 3


def test_nms_merge_empty():
    assert nms_merge([], [], []) == []
