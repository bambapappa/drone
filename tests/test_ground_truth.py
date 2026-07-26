"""Tests for ground-truth ("facit") scoring (Phase 5, report §5.6)."""

from __future__ import annotations

from review.ground_truth import score_ground_truth


def gt(aid, t, text="facit"):
    return {"annotation_id": aid, "t": t, "text": text}


def event(eid, t_start, category="STILLA"):
    return {"event_id": eid, "t_start": t_start, "t_end": t_start + 5.0, "category": category}


def note(aid, t, text="obs"):
    return {"annotation_id": aid, "t": t, "text": text}


def test_empty_everything():
    score = score_ground_truth([], [], [], tolerance_s=60)
    assert score.entries == []
    assert score.counts == {
        "gt_total": 0,
        "ai_found": 0,
        "ai_missed": 0,
        "operator_found": 0,
        "operator_missed": 0,
        "ai_unmatched": 0,
        "operator_unmatched": 0,
    }


def test_scores_both_sides_independently():
    gts = [gt("g1", 100.0), gt("g2", 400.0)]
    events = [event("e1", 110.0)]  # matches g1 only
    notes = [note("n1", 395.0)]  # matches g2 only
    score = score_ground_truth(gts, events, notes, tolerance_s=60)
    e1, e2 = score.entries
    assert e1["gt"]["annotation_id"] == "g1"
    assert e1["ai"]["event"]["event_id"] == "e1"
    assert e1["ai"]["delta_s"] == 10.0  # AI 10s after the reference time
    assert e1["operator"] is None
    assert e2["ai"] is None
    assert e2["operator"]["note"]["annotation_id"] == "n1"
    assert e2["operator"]["delta_s"] == -5.0  # operator noted 5s before
    assert score.counts["ai_found"] == 1
    assert score.counts["operator_found"] == 1
    assert score.counts["ai_missed"] == 1
    assert score.counts["operator_missed"] == 1


def test_one_to_one_assignment_prefers_nearest():
    gts = [gt("g1", 100.0), gt("g2", 130.0)]
    events = [event("e1", 128.0)]
    score = score_ground_truth(gts, events, [], tolerance_s=60)
    # e1 goes to g2 (|delta|=2 beats |delta|=28); g1 stays missed.
    assert score.entries[0]["ai"] is None
    assert score.entries[1]["ai"]["event"]["event_id"] == "e1"
    assert score.counts["ai_unmatched"] == 0


def test_unmatched_extras_reported():
    gts = [gt("g1", 100.0)]
    events = [event("e1", 100.0), event("e2", 500.0)]
    notes = [note("n1", 800.0)]
    score = score_ground_truth(gts, events, notes, tolerance_s=60)
    assert [e["event_id"] for e in score.ai_unmatched] == ["e2"]
    assert [n["annotation_id"] for n in score.operator_unmatched] == ["n1"]


def test_tolerance_bounds_matching():
    gts = [gt("g1", 100.0)]
    events = [event("e1", 170.0)]
    assert score_ground_truth(gts, events, [], tolerance_s=60).entries[0]["ai"] is None
    assert score_ground_truth(gts, events, [], tolerance_s=90).entries[0]["ai"] is not None


def test_deterministic_regardless_of_input_order():
    gts = [gt("g1", 100.0), gt("g2", 110.0)]
    events = [event("e1", 104.0), event("e2", 107.0)]
    a = score_ground_truth(gts, events, [], tolerance_s=60)
    b = score_ground_truth(list(reversed(gts)), list(reversed(events)), [], tolerance_s=60)
    assert a.entries == b.entries
    assert a.ai_unmatched == b.ai_unmatched


def test_entries_sorted_by_reference_time():
    gts = [gt("g2", 300.0), gt("g1", 10.0)]
    score = score_ground_truth(gts, [], [], tolerance_s=60)
    assert [e["gt"]["annotation_id"] for e in score.entries] == ["g1", "g2"]
