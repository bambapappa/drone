"""Tests for review.comparison — time-proximity matching between AI events
and imported operator field notes.

Covers the three-bucket split, the signed delta convention, nearest-match
tie-breaking when one side has multiple candidates, determinism regardless
of input order, and tolerance boundary behavior."""

from __future__ import annotations

import random

from review.comparison import compare_events_to_notes


def _event(event_id: str, t_start: float, t_end: float | None = None) -> dict:
    return {
        "event_id": event_id,
        "category": "STILLA",
        "person_id": 1,
        "t_start": t_start,
        "t_end": t_end if t_end is not None else t_start + 1.0,
        "confidence": 0.7,
        "evidence": {},
        "review": {"state": "unreviewed"},
    }


def _note(annotation_id: str, t: float, text: str = "observation") -> dict:
    return {"annotation_id": annotation_id, "t": t, "text": text, "raw_line": None}


class TestBasicMatching:
    def test_matches_within_tolerance(self):
        events = [_event("ev-1", 872.0)]
        notes = [_note("n-1", 870.0)]
        result = compare_events_to_notes(events, notes, tolerance_s=60.0)
        assert result.counts == {"both": 1, "ai_only": 0, "operator_only": 0}
        # AI's t_start (872) is AFTER the note's timestamp (870), i.e. the
        # operator noted it first -> negative delta.
        assert result.both[0].delta_s == -2.0

    def test_delta_sign_ai_detected_first(self):
        # Event at t=100, note at t=110 -> AI was 10s earlier -> delta positive.
        events = [_event("ev-1", 100.0)]
        notes = [_note("n-1", 110.0)]
        result = compare_events_to_notes(events, notes)
        assert result.both[0].delta_s == 10.0

    def test_no_match_beyond_tolerance(self):
        events = [_event("ev-1", 0.0)]
        notes = [_note("n-1", 200.0)]
        result = compare_events_to_notes(events, notes, tolerance_s=60.0)
        assert result.counts == {"both": 0, "ai_only": 1, "operator_only": 1}

    def test_exact_boundary_is_inclusive(self):
        events = [_event("ev-1", 0.0)]
        notes = [_note("n-1", 60.0)]
        result = compare_events_to_notes(events, notes, tolerance_s=60.0)
        assert result.counts["both"] == 1

    def test_just_beyond_boundary_excluded(self):
        events = [_event("ev-1", 0.0)]
        notes = [_note("n-1", 60.01)]
        result = compare_events_to_notes(events, notes, tolerance_s=60.0)
        assert result.counts["both"] == 0


class TestOneToOneAssignment:
    def test_event_prefers_nearest_note(self):
        events = [_event("ev-1", 100.0)]
        notes = [_note("n-far", 70.0), _note("n-near", 95.0)]
        result = compare_events_to_notes(events, notes, tolerance_s=60.0)
        assert result.counts["both"] == 1
        assert result.both[0].note["annotation_id"] == "n-near"
        assert result.counts["operator_only"] == 1
        assert result.operator_only[0]["annotation_id"] == "n-far"

    def test_note_prefers_nearest_event(self):
        events = [_event("ev-far", 40.0), _event("ev-near", 95.0)]
        notes = [_note("n-1", 100.0)]
        result = compare_events_to_notes(events, notes, tolerance_s=60.0)
        assert result.both[0].event["event_id"] == "ev-near"
        assert result.ai_only[0]["event_id"] == "ev-far"

    def test_greedy_global_assignment_not_purely_local(self):
        # ev-1 and note-1 are 5s apart (best mutual match); ev-2 is 10s from
        # note-1 and would also be a candidate, but note-1 is already taken by
        # the closer ev-1, so ev-2 falls back to note-2 (30s) instead of
        # going unmatched, since 30s is still within tolerance.
        events = [_event("ev-1", 100.0), _event("ev-2", 105.0)]
        notes = [_note("note-1", 95.0), _note("note-2", 135.0)]
        result = compare_events_to_notes(events, notes, tolerance_s=60.0)
        assert result.counts == {"both": 2, "ai_only": 0, "operator_only": 0}
        pairs = {(m.event["event_id"], m.note["annotation_id"]) for m in result.both}
        assert ("ev-1", "note-1") in pairs
        assert ("ev-2", "note-2") in pairs


class TestDeterminism:
    def test_result_independent_of_input_order(self):
        events = [_event(f"ev-{i}", float(i * 20)) for i in range(10)]
        notes = [_note(f"n-{i}", float(i * 20 + 3)) for i in range(10)]

        base = compare_events_to_notes(events, notes)
        shuffled_events = events[:]
        shuffled_notes = notes[:]
        random.Random(42).shuffle(shuffled_events)
        random.Random(7).shuffle(shuffled_notes)
        shuffled = compare_events_to_notes(shuffled_events, shuffled_notes)

        base_pairs = sorted((m.event["event_id"], m.note["annotation_id"]) for m in base.both)
        shuffled_pairs = sorted((m.event["event_id"], m.note["annotation_id"]) for m in shuffled.both)
        assert base_pairs == shuffled_pairs


class TestEmptyInputs:
    def test_no_events(self):
        result = compare_events_to_notes([], [_note("n-1", 10.0)])
        assert result.counts == {"both": 0, "ai_only": 0, "operator_only": 1}

    def test_no_notes(self):
        result = compare_events_to_notes([_event("ev-1", 10.0)], [])
        assert result.counts == {"both": 0, "ai_only": 1, "operator_only": 0}

    def test_both_empty(self):
        result = compare_events_to_notes([], [])
        assert result.counts == {"both": 0, "ai_only": 0, "operator_only": 0}


class TestOrdering:
    def test_both_sorted_by_note_time(self):
        events = [_event("ev-1", 200.0), _event("ev-2", 50.0)]
        notes = [_note("n-1", 205.0), _note("n-2", 48.0)]
        result = compare_events_to_notes(events, notes)
        assert [m.note["t"] for m in result.both] == [48.0, 205.0]

    def test_ai_only_sorted_by_t_start(self):
        events = [_event("ev-1", 500.0), _event("ev-2", 10.0)]
        result = compare_events_to_notes(events, [])
        assert [e["t_start"] for e in result.ai_only] == [10.0, 500.0]

    def test_operator_only_sorted_by_t(self):
        notes = [_note("n-1", 500.0), _note("n-2", 10.0)]
        result = compare_events_to_notes([], notes)
        assert [n["t"] for n in result.operator_only] == [10.0, 500.0]
