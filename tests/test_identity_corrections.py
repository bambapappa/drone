"""Tests for the identity-corrections projection (Phase 5 split/merge).

The projection replays live annotation ops over the engine's persons table
at read time — persons/<pass>.jsonl is never rewritten. These tests cover
the op semantics (merge, split), composition, undo-by-tombstone (an op list
without the undone op), graceful skipping of unresolvable ops, and the
determinism guarantee (same inputs → identical output).

Ops are keyed by the tracklet SET of each person they name, never by
person_id (which P3 assigns positionally and does not survive a
re-analysis) — the helpers below therefore derive the key from the person
records the op was recorded against, exactly like the write path does.
"""

from __future__ import annotations

from review.identity_corrections import (
    OLD_SCHEMA_REASON,
    STATE_MANUAL,
    apply_corrections,
    merge_overlap_warning,
)


def person(pid, tracklets, first=0.0, last=10.0, counts=None, centroids=None, audit=None, state="confirmed"):
    return {
        "person_id": pid,
        "tracklet_ids": list(tracklets),
        "embedding_centroids": centroids or {},
        "embedding_counts": counts or {},
        "first_seen": first,
        "last_seen": last,
        "confirmation_state": state,
        "assoc_audit": audit or [],
    }


def _key(persons, pid):
    """The tracklet-set key the UI would capture for `pid` at write time."""
    rec = next(p for p in persons if p["person_id"] == pid)
    return sorted(int(t) for t in rec["tracklet_ids"])


def op_merge(aid, ids, against=None, reason=None, member_keys=None):
    """A merge op as the write path records it: the members' full tracklet
    sets (from `against`, the projection at correction time) as the key,
    person_ids kept as provenance."""
    if member_keys is None:
        member_keys = [_key(against, p) for p in ids]
    return {
        "annotation_id": aid,
        "op": "merge",
        "person_ids": ids,
        "member_tracklet_ids": member_keys,
        "reason": reason,
    }


def op_split(aid, pid, tids, against=None, reason=None, source_key=None):
    if source_key is None:
        source_key = _key(against, pid)
    return {
        "annotation_id": aid,
        "op": "split",
        "person_id": pid,
        "tracklet_ids": tids,
        "source_tracklet_ids": source_key,
        "reason": reason,
    }


SPANS = {1: (0, 100), 2: (150, 300), 3: (400, 500), 4: (600, 700)}
FPS = 25.0


def test_no_ops_passthrough():
    persons = [person(1, [1]), person(2, [2])]
    res = apply_corrections(persons, [], fps=FPS)
    assert not res.changed
    assert [p["person_id"] for p in res.persons] == [1, 2]
    assert res.person_by_tracklet == {1: 1, 2: 2}
    # No correction markers on untouched persons.
    assert "corrected" not in res.persons[0]


def test_merge_two_persons():
    persons = [
        person(1, [1], first=0.0, last=4.0, counts={"hsv": 5}),
        person(2, [2], first=6.0, last=12.0, counts={"hsv": 9}, centroids={"hsv": [0.5, 0.5]}),
    ]
    res = apply_corrections(persons, [op_merge("aa", [1, 2], against=persons)], fps=FPS)
    assert res.applied == ["aa"]
    assert len(res.persons) == 1
    merged = res.persons[0]
    # Lowest id survives; spans widen to the union.
    assert merged["person_id"] == 1
    assert merged["tracklet_ids"] == [1, 2]
    assert merged["first_seen"] == 0.0
    assert merged["last_seen"] == 12.0
    # Centroids come from the member with most embedding samples (person 2).
    assert merged["embedding_centroids"] == {"hsv": [0.5, 0.5]}
    assert merged["confirmation_state"] == STATE_MANUAL
    assert merged["corrected"] is True
    assert merged["correction_ids"] == ["aa"]
    assert res.person_by_tracklet == {1: 1, 2: 1}


def test_split_detaches_tracklets_into_new_person():
    persons = [person(1, [1, 2], first=0.0, last=12.0), person(2, [3])]
    res = apply_corrections(persons, [op_split("bb", 1, [2], against=persons)], fps=FPS, tracklet_spans=SPANS)
    assert res.applied == ["bb"]
    by_id = {p["person_id"]: p for p in res.persons}
    # New person id = max existing + 1.
    assert set(by_id) == {1, 2, 3}
    assert by_id[1]["tracklet_ids"] == [1]
    assert by_id[3]["tracklet_ids"] == [2]
    # Spans recomputed from the tracklet frame spans.
    assert by_id[1]["first_seen"] == 0.0
    assert by_id[1]["last_seen"] == 100 / FPS
    assert by_id[3]["first_seen"] == 150 / FPS
    assert by_id[3]["last_seen"] == 300 / FPS
    # Split-off person has no fabricated appearance centroid.
    assert by_id[3]["embedding_centroids"] == {}
    assert by_id[3]["confirmation_state"] == STATE_MANUAL
    assert res.person_by_tracklet == {1: 1, 2: 3, 3: 2}


def test_ops_compose_in_append_order():
    # Split person 1, then merge the split-off with person 2.
    persons = [person(1, [1, 2]), person(2, [3])]
    # m1 names the split-off person 3, whose key is the state AFTER s1.
    ops = [op_split("s1", 1, [2], against=persons), op_merge("m1", [2, 3], member_keys=[[3], [2]])]
    res = apply_corrections(persons, ops, fps=FPS, tracklet_spans=SPANS)
    assert res.applied == ["s1", "m1"]
    by_id = {p["person_id"]: p for p in res.persons}
    assert set(by_id) == {1, 2}
    assert by_id[2]["tracklet_ids"] == [2, 3]


def test_undo_by_removing_op_from_live_list():
    # Tombstoning an op means it's simply absent from the live list —
    # the projection then reflects the remaining ops only.
    persons = [person(1, [1]), person(2, [2])]
    with_merge = apply_corrections(persons, [op_merge("aa", [1, 2], against=persons)], fps=FPS)
    without = apply_corrections(persons, [], fps=FPS)
    assert len(with_merge.persons) == 1
    assert len(without.persons) == 2


def test_unresolvable_op_is_skipped_and_reported():
    persons = [person(1, [1])]
    res = apply_corrections(persons, [op_merge("aa", [1, 99], member_keys=[[1], [99]])], fps=FPS)
    assert res.applied == []
    assert len(res.skipped) == 1
    assert res.skipped[0]["annotation_id"] == "aa"
    assert "99" in res.skipped[0]["reason"]
    # Untouched output.
    assert len(res.persons) == 1


def test_dependent_op_skipped_after_undo():
    # m1 merges 2+3 into 2; s1 then splits person 2. If m1 is undone,
    # s1 references tracklet 3 which person 2 no longer owns → skipped.
    persons = [person(2, [3]), person(3, [4])]
    ops_all = [
        op_merge("m1", [2, 3], against=persons),
        op_split("s1", 2, [4], source_key=[3, 4]),
    ]
    res_all = apply_corrections(persons, ops_all, fps=FPS, tracklet_spans=SPANS)
    assert res_all.applied == ["m1", "s1"]
    res_undone = apply_corrections(
        persons, [op_split("s1", 2, [4], source_key=[3, 4])], fps=FPS, tracklet_spans=SPANS
    )
    assert res_undone.applied == []
    assert res_undone.skipped[0]["annotation_id"] == "s1"


def test_split_rejects_all_tracklets_and_foreign_tracklets():
    persons = [person(1, [1, 2])]
    res = apply_corrections(persons, [op_split("x", 1, [1, 2], against=persons)], fps=FPS)
    assert res.applied == [] and "samtliga" in res.skipped[0]["reason"]
    res2 = apply_corrections(persons, [op_split("y", 1, [3], against=persons)], fps=FPS)
    assert res2.applied == [] and "tillhör inte" in res2.skipped[0]["reason"]


def test_merge_requires_two_distinct_existing():
    persons = [person(1, [1])]
    res = apply_corrections(persons, [op_merge("z", [1, 1], against=persons)], fps=FPS)
    assert res.applied == []


def test_projection_is_deterministic():
    persons = [person(1, [1, 2]), person(2, [3]), person(3, [4])]
    ops = [op_split("s", 1, [2], against=persons), op_merge("m", [2, 4], member_keys=[[3], [2]])]
    a = apply_corrections(persons, ops, fps=FPS, tracklet_spans=SPANS)
    b = apply_corrections(persons, ops, fps=FPS, tracklet_spans=SPANS)
    assert a.persons == b.persons
    assert a.person_by_tracklet == b.person_by_tracklet
    assert a.applied == b.applied


def test_engine_input_not_mutated():
    persons = [person(1, [1]), person(2, [2])]
    snapshot = [dict(p) for p in persons]
    apply_corrections(persons, [op_merge("aa", [1, 2], against=persons)], fps=FPS)
    assert persons == snapshot


# ---- keying: ops survive a renumber, or skip honestly ----
#
# P3 numbers persons positionally, so a re-analysis normally still produces
# contiguous ids 1..N — a person_id-keyed op would resolve and quietly act on
# a different pair of people. These tests pin the tracklet-set keying that
# makes the "skipped, never guessed at" promise true.

# Run A, the state the reviewer looked at when recording the corrections.
RUN_A = [person(1, [10, 11]), person(2, [20]), person(3, [30])]


def test_renumbered_run_with_same_topology_still_applies_correctly():
    # Same people, same tracklet sets, ids shuffled: person_id 2 now means
    # something else, but the recorded tracklet sets still identify the two
    # people the reviewer judged, so the merge lands on the right pair.
    run_b = [person(1, [30]), person(2, [10, 11]), person(3, [20])]
    op = op_merge("m", [2, 3], against=RUN_A)  # tracklets {20} + {30}
    res = apply_corrections(run_b, [op], fps=FPS)
    assert res.applied == ["m"]
    by_id = {p["person_id"]: p for p in res.persons}
    assert by_id[1]["tracklet_ids"] == [20, 30]
    assert by_id[1]["confirmation_state"] == STATE_MANUAL
    assert by_id[2]["tracklet_ids"] == [10, 11]  # untouched


def test_renumbered_run_with_changed_topology_skips_instead_of_wrong_pair():
    # Re-analysis grouped tracklets differently. Ids are STILL contiguous
    # 1..3, so a person_id-keyed replay would happily merge persons 2+3 and
    # split person 1 — both the wrong people. Keyed by tracklet set, neither
    # op resolves, so both are skipped and reported.
    run_b = [person(1, [10, 11, 12]), person(2, [20, 21]), person(3, [30])]
    ops = [
        op_merge("m", [2, 3], against=RUN_A),  # recorded key: {20} + {30}
        op_split("s", 1, [11], against=RUN_A),  # recorded key: {10, 11}
    ]
    res = apply_corrections(run_b, ops, fps=FPS, tracklet_spans=SPANS)

    assert res.applied == []
    assert [s["annotation_id"] for s in res.skipped] == ["m", "s"]
    for skip in res.skipped:
        assert "motsvarar inte längre exakt en person" in skip["reason"]
    # The wrong pair was NOT merged and no fabricated person appeared.
    assert [p["person_id"] for p in res.persons] == [1, 2, 3]
    assert [p["tracklet_ids"] for p in res.persons] == [[10, 11, 12], [20, 21], [30]]
    assert all(p["confirmation_state"] != STATE_MANUAL for p in res.persons)
    assert all(not p.get("corrected") for p in res.persons)


def test_missing_tracklets_named_in_the_skip_reason():
    run_b = [person(1, [10, 11]), person(2, [20])]  # tracklet 30 is gone
    res = apply_corrections(run_b, [op_merge("m", [2, 3], against=RUN_A)], fps=FPS)
    assert res.applied == []
    assert "spår saknas" in res.skipped[0]["reason"] and "30" in res.skipped[0]["reason"]


def test_engine_already_agrees_is_reported_as_skipped_not_applied():
    # The engine now associates exactly the tracklets the reviewer merged:
    # the judgment is already reality. Reported through the same skipped
    # list — there is no third bucket.
    run_b = [person(1, [10, 11]), person(2, [20, 30])]
    res = apply_corrections(run_b, [op_merge("m", [2, 3], against=RUN_A)], fps=FPS)
    assert res.applied == []
    assert res.skipped[0]["reason"] == "motorn associerar redan dessa tracklets som en person"
    assert run_b[1]["confirmation_state"] == "confirmed"


def test_old_schema_ops_are_skipped_never_backfilled():
    # Rows written before the tracklet-set key existed carry only positional
    # ids. They are skipped with a reason naming that cause — synthesising a
    # key from the current persons table would invent the evidence.
    legacy_merge = {"annotation_id": "old-m", "op": "merge", "person_ids": [1, 2]}
    legacy_split = {"annotation_id": "old-s", "op": "split", "person_id": 1, "tracklet_ids": [11]}
    res = apply_corrections(RUN_A, [legacy_merge, legacy_split], fps=FPS, tracklet_spans=SPANS)
    assert res.applied == []
    assert [s["reason"] for s in res.skipped] == [OLD_SCHEMA_REASON, OLD_SCHEMA_REASON]
    assert [p["tracklet_ids"] for p in res.persons] == [[10, 11], [20], [30]]


def test_merge_overlap_warning_fires_on_overlapping_spans():
    persons = {1: person(1, [1]), 2: person(2, [2])}
    # Overlapping spans → warning.
    spans = {1: (0, 200), 2: (150, 300)}
    warn = merge_overlap_warning(persons, [1, 2], spans)
    assert warn is not None and "samtidigt" in warn
    # Disjoint spans → no warning.
    assert merge_overlap_warning(persons, [1, 2], {1: (0, 100), 2: (150, 300)}) is None
    # No span data → no warning (can't tell).
    assert merge_overlap_warning(persons, [1, 2], None) is None
