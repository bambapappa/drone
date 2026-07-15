"""Tests for retroactive hazard-marker recompute (review.hazard).

Pure-logic tests over a synthetic sidecar (no torch/video/ML): seed P2
tracklets directly via ArtifactStore, then verify recompute_mot_fara derives
MOT_FARA against a manually placed danger point, and that it is
deterministic — moving the marker to the same position twice must produce
byte-identical events (report §5.1's "no re-running P1-P3" requirement
implies this is a pure query, not a batch job with its own variance).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.events import CATEGORY_MOT_FARA
from analysis.orchestrator import OfflineOrchestrator
from analysis.store import ArtifactStore
from review.hazard import recompute_mot_fara


@pytest.fixture
def store_with_tracklets(tmp_path: Path) -> ArtifactStore:
    """A run with P1 (fps) + P2 tracklets: one tracklet moving steadily in
    +x, fast enough and long enough to sustain MOT_FARA once a danger point
    is placed ahead of it."""
    store = ArtifactStore(str(tmp_path / "out"), "vh", "ch")
    store.create()
    store.record_pass_start(OfflineOrchestrator.P1_PASS_NAME, {"fps": 10.0})
    store.record_pass_complete(OfflineOrchestrator.P1_PASS_NAME, {})
    store.record_pass_start(OfflineOrchestrator.P2_PASS_NAME, {})
    store.start_fresh_pass_output("tracklets", OfflineOrchestrator.P2_PASS_NAME)
    for i in range(80):
        x = 50.0 + i * 4.0  # 40 px/s = 2x the 0.25 bh/s toward_speed default at 80px body
        store.add_tracklet_frame(
            OfflineOrchestrator.P2_PASS_NAME,
            tracklet_id=1,
            frame_no=i,
            det_id=i,
            data={"cls": "person", "conf": 0.9, "xyxy": [x, 100.0, x + 30.0, 180.0]},
        )
    store.record_pass_complete(OfflineOrchestrator.P2_PASS_NAME, {"total_tracklet_rows": 80})
    store.record_pass_start(OfflineOrchestrator.P5_PASS_NAME, {"config": {}})
    store.record_pass_complete(OfflineOrchestrator.P5_PASS_NAME, {})
    store.close()
    return ArtifactStore.open_readonly(store.run_dir)


class TestRecomputeMotFara:
    def test_danger_ahead_fires_mot_fara(self, store_with_tracklets: ArtifactStore):
        # Danger point far to the +x side, ahead of the subject's direction
        # of travel — should fire MOT_FARA.
        events = recompute_mot_fara(
            store_with_tracklets, person_by_tracklet={}, fps=10.0, hazard_x=2000.0, hazard_y=140.0
        )
        assert events
        assert all(e["category"] == CATEGORY_MOT_FARA for e in events)

    def test_danger_behind_does_not_fire_mot_fara(self, store_with_tracklets: ArtifactStore):
        # Danger point behind the subject (they're moving away from it) —
        # must not fire.
        events = recompute_mot_fara(
            store_with_tracklets, person_by_tracklet={}, fps=10.0, hazard_x=-2000.0, hazard_y=140.0
        )
        assert events == []

    def test_person_id_tagged_when_provided(self, store_with_tracklets: ArtifactStore):
        events = recompute_mot_fara(
            store_with_tracklets,
            person_by_tracklet={1: 9},
            fps=10.0,
            hazard_x=2000.0,
            hazard_y=140.0,
        )
        assert events and all(e["person_id"] == 9 for e in events)


class TestDeterminism:
    def test_same_position_twice_byte_identical(self, store_with_tracklets: ArtifactStore):
        def run():
            events = recompute_mot_fara(
                store_with_tracklets, person_by_tracklet={1: 9}, fps=10.0, hazard_x=2000.0, hazard_y=140.0
            )
            return json.dumps(events, sort_keys=True)

        assert run() == run()

    def test_moving_marker_away_and_back_reproduces_original(self, store_with_tracklets: ArtifactStore):
        # "moving the marker to the same position twice gives identical
        # results" — simulate a reviewer nudging the marker elsewhere and
        # then back.
        first = json.dumps(
            recompute_mot_fara(
                store_with_tracklets, person_by_tracklet={}, fps=10.0, hazard_x=2000.0, hazard_y=140.0
            ),
            sort_keys=True,
        )
        _ = recompute_mot_fara(
            store_with_tracklets, person_by_tracklet={}, fps=10.0, hazard_x=-500.0, hazard_y=400.0
        )
        back = json.dumps(
            recompute_mot_fara(
                store_with_tracklets, person_by_tracklet={}, fps=10.0, hazard_x=2000.0, hazard_y=140.0
            ),
            sort_keys=True,
        )
        assert first == back
