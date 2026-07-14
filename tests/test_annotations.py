"""Tests for the append-only annotation log (review.annotations).

Covers the Phase 2 surface (bookmarks + screenshots) and the invariants that
make the log safe against re-analysis destroying human review work:
  - append-only: deletes write tombstones, never rewrite
  - tombstones filter deleted entries on read
  - PNG bytes are stored under <run>/annotations/screenshots/, not in the JSONL
  - ids are unique (collision-resistant)
  - idempotent deletes (calling twice returns False the second time)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from review.annotations import AnnotationStore


@pytest.fixture
def store(tmp_path: Path) -> AnnotationStore:
    run_dir = tmp_path / "run-abc"
    run_dir.mkdir()
    s = AnnotationStore(run_dir)
    assert s.annotations_dir.exists()
    assert s.screenshots_dir.exists()
    return s


class TestBookmarks:
    def test_add_bookmark_returns_row_with_id(self, store: AnnotationStore):
        row = store.add_bookmark(t=12.5, label="Person vid bilen", note="oklart antal")
        assert "annotation_id" in row and len(row["annotation_id"]) == 12
        assert row["t"] == 12.5
        assert row["label"] == "Person vid bilen"
        assert row["note"] == "oklart antal"
        assert "created_at" in row

    def test_list_bookmarks_returns_live_only(self, store: AnnotationStore):
        store.add_bookmark(t=1.0, label="A")
        store.add_bookmark(t=2.0, label="B")
        rows = store.list_bookmarks()
        assert len(rows) == 2
        assert {r["label"] for r in rows} == {"A", "B"}

    def test_delete_bookmark_writes_tombstone(self, store: AnnotationStore):
        a = store.add_bookmark(t=1.0, label="A")
        b = store.add_bookmark(t=2.0, label="B")
        assert store.delete_bookmark(a["annotation_id"]) is True
        # A is tombstoned, B is still live.
        rows = store.list_bookmarks()
        assert len(rows) == 1
        assert rows[0]["annotation_id"] == b["annotation_id"]

    def test_delete_bookmark_idempotent(self, store: AnnotationStore):
        a = store.add_bookmark(t=1.0, label="A")
        assert store.delete_bookmark(a["annotation_id"]) is True
        # Second delete of the same id does nothing and reports False.
        assert store.delete_bookmark(a["annotation_id"]) is False

    def test_delete_unknown_id_is_noop(self, store: AnnotationStore):
        store.add_bookmark(t=1.0, label="A")
        assert store.delete_bookmark("nonexistent") is False
        assert len(store.list_bookmarks()) == 1

    def test_underlying_file_is_append_only(self, store: AnnotationStore):
        """Hard-deletes would rewrite the file; we must never do that. The
        raw file should contain: 2 row entries + 1 tombstone after one delete."""
        a = store.add_bookmark(t=1.0, label="A")
        store.add_bookmark(t=2.0, label="B")
        store.delete_bookmark(a["annotation_id"])
        raw = (store.annotations_dir / "bookmarks.jsonl").read_text().strip().splitlines()
        assert len(raw) == 3  # A, B, tombstone(A)
        kinds = [json.loads(line).get("action") or "row" for line in raw]
        assert kinds.count("row") == 2
        assert kinds.count("delete") == 1


class TestScreenshots:
    def test_add_screenshot_metadata_only(self, store: AnnotationStore):
        row = store.add_screenshot(t=42.0, label="Skärmdump 1")
        assert row["t"] == 42.0
        assert row["label"] == "Skärmdump 1"
        assert row["png_filename"] is None

    def test_add_screenshot_with_png_stores_file(self, store: AnnotationStore):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64  # not a real PNG, just bytes
        row = store.add_screenshot(t=42.0, label="Skärmdump 1", png_bytes=png_bytes)
        assert row["png_filename"] is not None
        assert row["png_filename"].endswith(".png")
        path = store.screenshots_dir / row["png_filename"]
        assert path.exists()
        assert path.read_bytes() == png_bytes

    def test_screenshot_png_path_resolves(self, store: AnnotationStore):
        row = store.add_screenshot(t=42.0, label="x", png_bytes=b"PNG-CONTENT")
        resolved = store.screenshot_png_path(row["annotation_id"])
        assert resolved is not None
        assert resolved.read_bytes() == b"PNG-CONTENT"

    def test_screenshot_png_path_none_when_metadata_only(self, store: AnnotationStore):
        row = store.add_screenshot(t=42.0, label="x")
        assert store.screenshot_png_path(row["annotation_id"]) is None

    def test_screenshot_png_path_none_for_unknown_id(self, store: AnnotationStore):
        assert store.screenshot_png_path("nonexistent") is None

    def test_delete_screenshot_keeps_png_on_disk(self, store: AnnotationStore):
        """Tombstones suppress the row but do not GC the PNG — disk GC is
        not the log's job, and an undelete could still want the bytes."""
        row = store.add_screenshot(t=42.0, label="x", png_bytes=b"PNG")
        path = store.screenshots_dir / row["png_filename"]
        assert path.exists()
        assert store.delete_screenshot(row["annotation_id"]) is True
        # File remains on disk...
        assert path.exists()
        # ...but the metadata row is gone.
        assert all(r["annotation_id"] != row["annotation_id"] for r in store.list_screenshots())


class TestVerdicts:
    """Confirm/reject/note on an event_id. Unlike bookmarks/screenshots,
    verdicts are keyed by event_id with latest-row-wins semantics — there is
    no delete, only new state transitions appended over time."""

    def test_first_verdict_defaults_missing_fields(self, store: AnnotationStore):
        row = store.set_verdict("stilla-000001", state="confirmed")
        assert row["event_id"] == "stilla-000001"
        assert row["state"] == "confirmed"
        assert row["note"] is None
        assert row["reviewer"] is None
        assert "reviewed_at" not in row or "created_at" in row

    def test_get_verdict_returns_none_when_unreviewed(self, store: AnnotationStore):
        assert store.get_verdict("nonexistent") is None

    def test_note_only_update_carries_forward_state(self, store: AnnotationStore):
        store.set_verdict("ev-1", state="confirmed", reviewer="Anna")
        row = store.set_verdict("ev-1", note="ser ut som en figurant")
        assert row["state"] == "confirmed"
        assert row["reviewer"] == "Anna"
        assert row["note"] == "ser ut som en figurant"

    def test_state_change_carries_forward_note(self, store: AnnotationStore):
        store.set_verdict("ev-1", state="confirmed", note="ok")
        row = store.set_verdict("ev-1", state="rejected")
        assert row["state"] == "rejected"
        assert row["note"] == "ok"

    def test_get_verdict_returns_latest_of_several_transitions(self, store: AnnotationStore):
        store.set_verdict("ev-1", state="confirmed")
        store.set_verdict("ev-1", state="rejected")
        store.set_verdict("ev-1", state="confirmed", note="ombedömd")
        row = store.get_verdict("ev-1")
        assert row["state"] == "confirmed"
        assert row["note"] == "ombedömd"

    def test_rejects_unknown_state(self, store: AnnotationStore):
        with pytest.raises(ValueError):
            store.set_verdict("ev-1", state="maybe")

    def test_all_verdicts_keeps_only_latest_per_event(self, store: AnnotationStore):
        store.set_verdict("ev-1", state="confirmed")
        store.set_verdict("ev-2", state="rejected")
        store.set_verdict("ev-1", state="rejected")
        latest = store.all_verdicts()
        assert set(latest.keys()) == {"ev-1", "ev-2"}
        assert latest["ev-1"]["state"] == "rejected"
        assert latest["ev-2"]["state"] == "rejected"

    def test_verdict_log_is_append_only(self, store: AnnotationStore):
        """Every set_verdict call appends a row — the full transition
        history must be reconstructable, not overwritten in place."""
        store.set_verdict("ev-1", state="confirmed")
        store.set_verdict("ev-1", state="rejected")
        raw = (store.annotations_dir / "verdicts.jsonl").read_text().strip().splitlines()
        assert len(raw) == 2
        states = [json.loads(line)["state"] for line in raw]
        assert states == ["confirmed", "rejected"]


class TestOperatorNotes:
    def test_add_operator_note_returns_row_with_id(self, store: AnnotationStore):
        row = store.add_operator_note(
            t=872.0, text="2 personer vid fordonet", raw_line="2 personer vid fordonet, 14:32"
        )
        assert "annotation_id" in row
        assert row["t"] == 872.0
        assert row["text"] == "2 personer vid fordonet"
        assert row["raw_line"] == "2 personer vid fordonet, 14:32"

    def test_list_operator_notes_returns_live_only(self, store: AnnotationStore):
        store.add_operator_note(t=1.0, text="a")
        store.add_operator_note(t=2.0, text="b")
        assert len(store.list_operator_notes()) == 2

    def test_delete_operator_note_tombstones(self, store: AnnotationStore):
        a = store.add_operator_note(t=1.0, text="a")
        store.add_operator_note(t=2.0, text="b")
        assert store.delete_operator_note(a["annotation_id"]) is True
        rows = store.list_operator_notes()
        assert len(rows) == 1
        assert rows[0]["text"] == "b"

    def test_delete_operator_note_idempotent(self, store: AnnotationStore):
        a = store.add_operator_note(t=1.0, text="a")
        assert store.delete_operator_note(a["annotation_id"]) is True
        assert store.delete_operator_note(a["annotation_id"]) is False


class TestBulk:
    def test_all_annotations_groups_by_kind(self, store: AnnotationStore):
        store.add_bookmark(t=1.0, label="b1")
        store.add_screenshot(t=2.0, label="s1")
        store.add_operator_note(t=3.0, text="n1")
        payload = store.all_annotations()
        assert set(payload.keys()) == {"bookmarks", "screenshots", "operator_notes"}
        assert len(payload["bookmarks"]) == 1
        assert len(payload["screenshots"]) == 1
        assert len(payload["operator_notes"]) == 1

    def test_export_payload_returns_live_entries(self, store: AnnotationStore):
        a = store.add_bookmark(t=1.0, label="b1")
        store.add_screenshot(t=2.0, label="s1")
        store.add_operator_note(t=3.0, text="n1")
        store.delete_bookmark(a["annotation_id"])
        payload = store.export_payload()
        assert payload["bookmarks"] == []
        assert len(payload["screenshots"]) == 1
        assert len(payload["operator_notes"]) == 1


class TestIsolationFromEngine:
    """The annotation log MUST be untouched by an engine re-run. This test
    simulates the scenario: write annotations, then have the engine rewrite
    an adjacent AI table (events/) — annotations must survive."""

    def test_annotations_survive_events_rewrite(self, tmp_path: Path):
        from analysis.store import ArtifactStore

        # Create an engine-style run.
        store = ArtifactStore(str(tmp_path), "vh", "ch")
        store.create()
        # Annotations written by the review layer.
        ann = AnnotationStore(store.run_dir)
        ann.add_bookmark(t=5.0, label="checkpoint")
        # Engine rewrites events/ (start_fresh_pass_output truncates).
        store.start_fresh_pass_output("events", "p5_events")
        store.add_event(
            "p5_events",
            "ev-1",
            {
                "category": "STILLA",
                "person_id": None,
                "t_start": 0.0,
                "t_end": 1.0,
                "confidence": 0.5,
                "evidence": {},
                "review": {"state": "unreviewed"},
            },
        )
        # Annotations are intact.
        ann2 = AnnotationStore(store.run_dir)
        rows = ann2.list_bookmarks()
        assert len(rows) == 1
        assert rows[0]["label"] == "checkpoint"
