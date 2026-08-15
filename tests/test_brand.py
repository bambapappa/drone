"""Persistent, place-bound fire incidents built from fire/smoke observations."""

from analysis.brand import BrandIncidentTracker, BrandObservation


def obs(frame_no, x, y, *, signal="smoke", segment=1, area=0.01):
    return BrandObservation(
        frame_no=frame_no,
        signal=signal,
        pos=(x, y),
        area=area,
        scene_segment=segment,
    )


def test_fire_and_smoke_at_same_place_become_one_brand():
    tracker = BrandIncidentTracker(association_radius=20.0)

    tracker.observe([obs(10, 100.0, 100.0, signal="smoke")])
    tracker.observe([obs(20, 108.0, 96.0, signal="fire")])

    incidents = tracker.incidents(end_frame=99, fps=10.0)
    assert len(incidents) == 1
    assert incidents[0].signals == frozenset({"fire", "smoke"})
    assert incidents[0].frame_start == 10
    assert incidents[0].frame_end == 99
    assert incidents[0].last_observed_frame == 20
    assert incidents[0].observed_frame_count == 2


def test_spatially_separate_observations_remain_separate_brands():
    tracker = BrandIncidentTracker(association_radius=20.0)

    tracker.observe(
        [
            obs(10, 100.0, 100.0),
            obs(10, 300.0, 300.0),
        ]
    )

    incidents = tracker.incidents(end_frame=99, fps=10.0)
    assert [incident.brand_id for incident in incidents] == ["brand-000000", "brand-000001"]
    assert [incident.anchor for incident in incidents] == [(100.0, 100.0), (300.0, 300.0)]


def test_brand_stays_active_through_observation_gaps_until_video_end():
    tracker = BrandIncidentTracker(association_radius=20.0)
    tracker.observe([obs(10, 100.0, 100.0)])

    assert tracker.danger_targets(frame_no=500, scene_segment=1) == {"brand-000000": (100.0, 100.0)}
    assert tracker.incidents(end_frame=999, fps=10.0)[0].t_end == 100.0


def test_unlinked_scene_segments_are_never_automatically_merged():
    tracker = BrandIncidentTracker(association_radius=20.0)

    tracker.observe([obs(10, 100.0, 100.0, segment=1)])
    tracker.observe([obs(20, 101.0, 101.0, segment=2)])

    incidents = tracker.incidents(end_frame=99, fps=10.0)
    assert len(incidents) == 2
    assert tracker.danger_targets(50, scene_segment=1) == {"brand-000000": (100.0, 100.0)}
    assert tracker.danger_targets(50, scene_segment=2) == {"brand-000001": (101.0, 101.0)}


def test_each_observation_can_update_only_one_incident():
    tracker = BrandIncidentTracker(association_radius=20.0)
    tracker.observe([obs(10, 100.0, 100.0), obs(10, 130.0, 100.0)])

    tracker.observe([obs(20, 115.0, 100.0, signal="fire")])

    incidents = tracker.incidents(end_frame=99, fps=10.0)
    counts = sorted(incident.observation_count for incident in incidents)
    assert counts == [1, 2]


def test_fire_and_smoke_in_same_frame_count_as_one_observed_frame():
    tracker = BrandIncidentTracker(association_radius=20.0)
    tracker.observe(
        [
            obs(10, 100.0, 100.0, signal="fire"),
            obs(10, 101.0, 100.0, signal="smoke"),
        ]
    )

    incident = tracker.incidents(end_frame=99, fps=10.0)[0]
    assert incident.observation_count == 2
    assert incident.observed_frame_count == 1


def test_single_signal_jump_updates_existing_site_instead_of_inventing_another():
    tracker = BrandIncidentTracker(association_radius=20.0)
    tracker.observe([obs(10, 100.0, 100.0)])

    tracker.observe([obs(20, 400.0, 400.0)])

    incidents = tracker.incidents(end_frame=99, fps=10.0)
    assert len(incidents) == 1
    assert incidents[0].observation_count == 2


def test_second_site_requires_simultaneous_same_signal_evidence():
    tracker = BrandIncidentTracker(association_radius=20.0)
    tracker.observe([obs(10, 100.0, 100.0), obs(10, 400.0, 400.0)])

    incidents = tracker.incidents(end_frame=99, fps=10.0)
    assert len(incidents) == 2
