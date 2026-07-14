"""Tests for P3 identity: global tracklet association into persons.

These are the Phase 1 acceptance tests for the identity design (architecture
report §3): occlusion re-entry re-associates correctly, the hard temporal-
overlap exclusion keeps two simultaneously-visible same-appearance tracklets
apart, the spatio-temporal and appearance gates fire as specified, the audit
trail records every decision, the count is honestly uncertainty-banded, and
the whole thing is deterministic.

Pure-logic tests (no torch, no model weights): TrackletProfiles are built
directly, the same way run_pass_p3 builds them from P1+P2 output. The
end-to-end P3 determinism test (byte-identical persons/assoc_audit across two
runs) lives in test_orchestrator_p3.py.
"""

from __future__ import annotations

import copy

import numpy as np

from analysis.identity import (
    TrackletProfile,
    associate,
    build_tracklet_profiles,
)
from analysis.orchestrator import OfflineConfig

FPS = 10.0


def _profile(
    tid: int,
    f0: int,
    f1: int,
    emb: list[float] | None = None,
    method: str = "osnet",
    start_center: tuple[float, float] = (100.0, 100.0),
    end_center: tuple[float, float] | None = None,
    counts: int | None = None,
) -> TrackletProfile:
    centroids: dict[str, np.ndarray] = {}
    cts: dict[str, int] = {}
    if emb is not None:
        v = np.asarray(emb, dtype=np.float64)
        v = v / max(np.linalg.norm(v), 1e-12)
        centroids[method] = v
        cts[method] = counts if counts is not None else (f1 - f0 + 1)
    return TrackletProfile(
        tracklet_id=tid,
        cls="person",
        frame_start=f0,
        frame_end=f1,
        first_seen=f0 / FPS,
        last_seen=f1 / FPS,
        frames=set(range(f0, f1 + 1)),
        centroids=centroids,
        counts=cts,
        start_center=start_center,
        end_center=end_center if end_center is not None else start_center,
    )


RED = [1.0, 0.0, 0.0]  # same-appearance "person in red"
BLUE = [0.0, 1.0, 0.0]
DIAG = 500.0


class TestOcclusionReentry:
    """The Phase 1 exit criterion: a person who leaves and re-enters (tracklet
    dies, new tracklet appears after a gap) is re-associated into one person."""

    def test_split_tracklet_merges_across_gap(self):
        cfg = OfflineConfig()
        # Tracklet 1: frames 0-9 at x~50. Tracklet 2: frames 20-29 at x~56,
        # same appearance, 1.1 s gap, spatially close -> one person.
        a = _profile(1, 0, 9, RED, start_center=(50, 100), end_center=(55, 100))
        b = _profile(2, 20, 29, RED, start_center=(56, 100), end_center=(60, 100))
        res = associate([a, b], cfg, DIAG)
        assert len(res.persons) == 1
        assert res.persons[0].tracklet_ids == [1, 2]
        # spans 0..29 frames = 2.9 s >= confirm_s -> confirmed
        assert res.persons[0].confirmation_state == "confirmed"

    def test_merge_audit_records_gate_values(self):
        cfg = OfflineConfig()
        a = _profile(1, 0, 9, RED, start_center=(50, 100), end_center=(55, 100))
        b = _profile(2, 20, 29, RED, start_center=(56, 100), end_center=(60, 100))
        res = associate([a, b], cfg, DIAG)
        audit = res.persons[0].assoc_audit
        merges = [e for e in audit if e["rule"] == "merged"]
        assert len(merges) == 1
        m = merges[0]
        assert m["tracklet_a"] == 1 and m["tracklet_b"] == 2
        assert m["appearance_sim"] > cfg.p3_sim_thresh  # passed the appearance gate
        assert m["gap_s"] > 0  # a real re-entry gap
        # dist (55->56 = 1 px) must be under the limit
        assert m["dist"] <= m["dist_limit"]


class TestTemporalOverlapExclusion:
    """Two tracklets visible in the same frame are NEVER the same person — the
    hard offline-only constraint (impossible live, trivial offline)."""

    def test_overlapping_same_appearance_stays_separate(self):
        cfg = OfflineConfig()
        # Both present in frames 0-9, identical appearance, different positions.
        a = _profile(1, 0, 9, RED, start_center=(50, 100), end_center=(50, 100))
        b = _profile(2, 0, 9, RED, start_center=(400, 100), end_center=(400, 100))
        res = associate([a, b], cfg, DIAG)
        assert len(res.persons) == 2
        # Same-clothing twins seen together is the B4 ambiguity -> uncertainty band
        assert res.uncertain_merges >= 1

    def test_partially_overlapping_excluded(self):
        cfg = OfflineConfig()
        # Share only frame 5 -> still excluded (one shared frame is enough).
        a = _profile(1, 0, 5, RED, start_center=(50, 100), end_center=(55, 100))
        b = _profile(2, 5, 15, RED, start_center=(56, 100), end_center=(60, 100))
        res = associate([a, b], cfg, DIAG)
        assert len(res.persons) == 2  # NOT merged despite identical appearance


class TestSpatioTemporalGate:
    def test_too_far_reentry_blocked(self):
        cfg = OfflineConfig(p3_max_gap_s=60.0)
        # Tiny gap but person "teleported" across the frame -> implausible.
        a = _profile(1, 0, 9, RED, start_center=(50, 100), end_center=(50, 100))
        b = _profile(2, 10, 19, RED, start_center=(490, 100), end_center=(490, 100))
        res = associate([a, b], cfg, DIAG)
        assert len(res.persons) == 2  # blocked by spatial gate

    def test_far_reentry_allowed_after_long_gap(self):
        # The (1+gap) term relaxes the gate with time: after a long gap the
        # spatial gate no longer binds and appearance decides (design intent).
        cfg = OfflineConfig(p3_max_gap_s=120.0)
        a = _profile(1, 0, 9, RED, start_center=(50, 100), end_center=(50, 100))
        # gap ~ 50s (frames 500..509) -> dist_limit = 0.45*500*(1+50) huge
        b = _profile(2, 500, 509, RED, start_center=(490, 100), end_center=(490, 100))
        res = associate([a, b], cfg, DIAG)
        assert len(res.persons) == 1  # merged: long gap relaxes the spatial gate

    def test_max_gap_blocks_too_long_reentry(self):
        cfg = OfflineConfig(p3_max_gap_s=5.0)
        a = _profile(1, 0, 9, RED, start_center=(50, 100), end_center=(50, 100))
        b = _profile(2, 100, 109, RED, start_center=(55, 100), end_center=(60, 100))
        res = associate([a, b], cfg, DIAG)
        assert len(res.persons) == 2  # 9s gap > max_gap_s -> blocked


class TestAppearanceGate:
    def test_different_appearance_not_merged(self):
        cfg = OfflineConfig()
        a = _profile(1, 0, 9, RED, start_center=(50, 100), end_center=(55, 100))
        b = _profile(2, 20, 29, BLUE, start_center=(56, 100), end_center=(60, 100))
        res = associate([a, b], cfg, DIAG)
        assert len(res.persons) == 2

    def test_mixed_methods_not_compared(self):
        # An osnet vector and an hsv vector are different-dimensional / not
        # comparable: no appearance evidence -> never merged on appearance.
        cfg = OfflineConfig()
        a = _profile(1, 0, 9, [1.0, 0.0], method="osnet", start_center=(50, 100), end_center=(55, 100))
        b = _profile(2, 20, 29, [1.0, 0.0], method="hsv", start_center=(56, 100), end_center=(60, 100))
        res = associate([a, b], cfg, DIAG)
        assert len(res.persons) == 2


class TestHonestCounting:
    def test_uncertainty_band_counts_near_merges(self):
        # Appearance just below threshold but otherwise feasible -> uncertain.
        cfg = OfflineConfig(p3_sim_thresh=0.90, p3_uncertain_margin=0.10)
        # sim = 0.5 (orthogonal-ish) is below 0.90-0.10=0.80 -> NOT in band.
        # Use a vector at ~0.85 cosine to land in [0.80, 0.90).
        import math

        ang = math.acos(0.85)
        emb_b = [math.cos(ang), math.sin(ang)]
        a = _profile(1, 0, 9, [1.0, 0.0], start_center=(50, 100), end_center=(55, 100))
        b = _profile(2, 20, 29, emb_b, start_center=(56, 100), end_center=(60, 100))
        res = associate([a, b], cfg, DIAG)
        assert len(res.persons) == 2  # not merged (below thresh)
        assert res.uncertain_merges == 1  # but in the honesty band

    def test_confirmed_count_excludes_transient(self):
        cfg = OfflineConfig(p3_confirm_s=2.0)
        # A 1-frame tracklet: 0.1s span -> transient, not counted as confirmed.
        a = _profile(1, 0, 0, RED, start_center=(50, 100), end_center=(50, 100))
        # A long tracklet -> confirmed.
        b = _profile(2, 0, 50, BLUE, start_center=(300, 100), end_center=(300, 100))
        res = associate([a, b], cfg, DIAG)
        assert res.confirmed_count == 1
        states = {p.confirmation_state for p in res.persons}
        assert "transient" in states and "confirmed" in states


class TestDeterminism:
    def _run(self):
        cfg = OfflineConfig()
        profs = [
            _profile(1, 0, 9, RED, start_center=(50, 100), end_center=(55, 100)),
            _profile(2, 20, 29, RED, start_center=(56, 100), end_center=(60, 100)),
            _profile(3, 0, 9, BLUE, start_center=(300, 100), end_center=(300, 100)),
        ]
        return associate(profs, cfg, DIAG)

    def test_two_runs_identical(self):
        r1 = self._run()
        r2 = self._run()
        # Person ids, tracklet membership, centroids, audit all identical.
        assert len(r1.persons) == len(r2.persons)
        for p1, p2 in zip(r1.persons, r2.persons):
            assert p1.person_id == p2.person_id
            assert p1.tracklet_ids == p2.tracklet_ids
            assert p1.assoc_audit == p2.assoc_audit
            assert p1.confirmation_state == p2.confirmation_state
        assert r1.confirmed_count == r2.confirmed_count
        assert r1.uncertain_merges == r2.uncertain_merges

    def test_input_order_does_not_change_result(self):
        """Agglomerative clustering must be order-independent: shuffling the
        input tracklet order must not change the final persons (global
        association, not greedy-in-input-order)."""
        cfg = OfflineConfig()
        base = [
            _profile(1, 0, 9, RED, start_center=(50, 100), end_center=(55, 100)),
            _profile(2, 20, 29, RED, start_center=(56, 100), end_center=(60, 100)),
            _profile(3, 0, 9, BLUE, start_center=(300, 100), end_center=(300, 100)),
        ]
        shuffled = copy.deepcopy(base)[::-1]
        r1 = associate(base, cfg, DIAG)
        r2 = associate(shuffled, cfg, DIAG)
        ids1 = {tuple(p.tracklet_ids) for p in r1.persons}
        ids2 = {tuple(p.tracklet_ids) for p in r2.persons}
        assert ids1 == ids2


class TestPersonIdOrdering:
    def test_ids_assigned_by_first_seen_then_tracklet(self):
        cfg = OfflineConfig()
        # Three distinct people; ensure deterministic id assignment.
        late = _profile(5, 100, 109, BLUE, start_center=(300, 100), end_center=(300, 100))
        early = _profile(2, 0, 50, RED, start_center=(50, 100), end_center=(50, 100))
        mid = _profile(7, 50, 99, [0.0, 0.0, 1.0], start_center=(200, 100), end_center=(200, 100))
        res = associate([late, early, mid], cfg, DIAG)
        # person 1 = earliest first_seen (early, t=0); then mid (t=5); then late (t=10)
        assert res.persons[0].tracklet_ids == [2]
        assert res.persons[1].tracklet_ids == [7]
        assert res.persons[2].tracklet_ids == [5]
        assert [p.person_id for p in res.persons] == [1, 2, 3]


class TestBuildTrackletProfiles:
    """The join from P2 tracklet rows + P1 detection records (via det_id) into
    TrackletProfiles — exercised at the data level run_pass_p3 uses."""

    def test_join_aggregates_embeddings_per_tracklet(self):
        rows = [
            {"tracklet_id": 1, "frame_no": 0, "det_id": 10, "cls": "person", "xyxy": [0, 0, 20, 40]},
            {"tracklet_id": 1, "frame_no": 1, "det_id": 11, "cls": "person", "xyxy": [2, 0, 22, 40]},
            {"tracklet_id": 2, "frame_no": 0, "det_id": 12, "cls": "person", "xyxy": [200, 0, 220, 40]},
        ]
        dets = {
            10: {"embedding": [1.0, 0.0], "embedding_method": "osnet"},
            11: {"embedding": [1.0, 0.0], "embedding_method": "osnet"},
            12: {"embedding": [0.0, 1.0], "embedding_method": "osnet"},
        }
        profs = build_tracklet_profiles(rows, dets, FPS)
        by_id = {p.tracklet_id: p for p in profs}
        # tracklet 1 aggregated two detections into one osnet centroid
        assert by_id[1].counts["osnet"] == 2
        assert by_id[1].centroids["osnet"].tolist() == [1.0, 0.0]
        assert by_id[2].counts["osnet"] == 1
        # start/end centers from min/max frame
        assert by_id[1].start_center == (10.0, 20.0)  # frame 0 center
        assert by_id[1].end_center == (12.0, 20.0)  # frame 1 center

    def test_missing_embedding_skipped(self):
        rows = [{"tracklet_id": 1, "frame_no": 0, "det_id": 0, "cls": "person", "xyxy": [0, 0, 10, 10]}]
        dets = {0: {"embedding": None, "embedding_method": None}}
        profs = build_tracklet_profiles(rows, dets, FPS)
        assert profs[0].centroids == {}  # no embedding -> no centroid

    def test_sorted_by_tracklet_id(self):
        rows = [
            {"tracklet_id": 3, "frame_no": 0, "det_id": 0, "cls": "person", "xyxy": [0, 0, 10, 10]},
            {"tracklet_id": 1, "frame_no": 0, "det_id": 1, "cls": "person", "xyxy": [0, 0, 10, 10]},
        ]
        profs = build_tracklet_profiles(rows, {}, FPS)
        assert [p.tracklet_id for p in profs] == [1, 3]
