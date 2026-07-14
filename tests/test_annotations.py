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


class TestBulk:
    def test_all_annotations_groups_by_kind(self, store: AnnotationStore):
        store.add_bookmark(t=1.0, label="b1")
        store.add_screenshot(t=2.0, label="s1")
        payload = store.all_annotations()
        assert set(payload.keys()) == {"bookmarks", "screenshots"}
        assert len(payload["bookmarks"]) == 1
        assert len(payload["screenshots"]) == 1

    def test_export_payload_returns_live_entries(self, store: AnnotationStore):
        a = store.add_bookmark(t=1.0, label="b1")
        store.add_screenshot(t=2.0, label="s1")
        store.delete_bookmark(a["annotation_id"])
        payload = store.export_payload()
        assert payload["bookmarks"] == []
        assert len(payload["screenshots"]) == 1


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
