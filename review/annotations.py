"""Append-only annotation log for human review data.

Lives alongside the AI-generated tables in the sidecar but is intentionally
a separate persistence layer:

- The AI tables (frames/, detections/, tracklets/, persons/, events/) are
  written once by the engine and rewritten on every re-analysis — that is
  the point of "reproducible re-runs" (re-run = byte-identical output).
- The annotations layer is **append-only** and survives re-analysis: a
  reviewer's bookmarks, screenshots, confirm/reject verdicts, and identity
  corrections accumulate across review sessions and are never silently
  destroyed by a fresh engine run (architecture report §2.4 — annotations
  are a separate log keyed to artifact version).

Implementation: one JSONL file per annotation kind under
`<run_id>/annotations/`. Each row is either a record or a tombstone
(`{"action": "delete", "target_id": ...}`) — hard deletes would violate
append-only; tombstones preserve the audit trail while letting the read
path filter out deleted entries.

Phase 2 implements two kinds:
  - bookmarks:   {t, label, note} — reviewer's named time markers
  - screenshots: {t, label, note, png_filename?} — link to a client-composited
                 PNG saved to disk (the compositing itself happens in the
                 browser; this module never renders a frame — that is the
                 "one annotated-frame renderer" rule from report §2.5)

Phase 3 will add (without restructuring this module):
  - verdicts:    confirm/reject/note on a specific event_id
  - identity corrections: split/merge person_id at time t with a reason
  - operator-notes import: timestamped rows from the live session
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BOOKMARKS = "bookmarks"
SCREENSHOTS = "screenshots"
PHASE2_KINDS = (BOOKMARKS, SCREENSHOTS)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _annotation_id() -> str:
    """Random 12-char hex id. ids are not deterministic — annotations are
    human-written at human-clicked moments, so determinism does not apply
    (unlike the engine's stable event_id space). Uniqueness is what matters;
    12 chars of secrets.token_hex gives 48 bits of entropy, which is plenty
    for collision resistance within a single run."""
    return secrets.token_hex(6)


class AnnotationStore:
    """Append-only log for human review data, layered over an analysis run.

    Constructed from a run_dir (the same directory ArtifactStore owns).
    Reads filter out tombstones so callers see only live entries; writes
    append a row (create) or a tombstone (delete). Never rewrites existing
    rows — the audit trail is the point.
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.annotations_dir = self.run_dir / "annotations"
        self.screenshots_dir = self.annotations_dir / "screenshots"
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        # Screenshots PNGs live in a subdir; their metadata is in the JSONL.
        self.screenshots_dir.mkdir(exist_ok=True)

    # ---- path helpers ----

    def _log_path(self, kind: str) -> Path:
        if kind not in PHASE2_KINDS:
            raise ValueError(f"unknown annotation kind: {kind!r}")
        return self.annotations_dir / f"{kind}.jsonl"

    # ---- generic append + read ----

    def _append(self, kind: str, row: dict[str, Any]) -> dict[str, Any]:
        """Append one row to <kind>.jsonl. Returns the row with id/created_at."""
        row = dict(row)
        row.setdefault("annotation_id", _annotation_id())
        row.setdefault("created_at", _utc_now())
        path = self._log_path(kind)
        with open(path, "a") as f:
            f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        return row

    def _append_tombstone(self, kind: str, annotation_id: str) -> None:
        """Append a soft-delete marker. The read path filters out both the
        tombstone and the entry it targets. Hard-delete would rewrite the
        file and lose the audit trail — a no-go for an append-only log."""
        row = {"action": "delete", "target_id": annotation_id, "deleted_at": _utc_now()}
        with open(self._log_path(kind), "a") as f:
            f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")

    def _iter_raw(self, kind: str) -> Iterable[dict[str, Any]]:
        path = self._log_path(kind)
        if not path.exists():
            return
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _live_rows(self, kind: str) -> list[dict[str, Any]]:
        """All non-tombstoned rows in append order. O(rows) — fine for the
        review use case (annotation counts stay small per run)."""
        deleted: set[str] = set()
        rows: list[dict[str, Any]] = []
        for row in self._iter_raw(kind):
            if row.get("action") == "delete":
                deleted.add(row.get("target_id"))
                continue
            rows.append(row)
        return [r for r in rows if r.get("annotation_id") not in deleted]

    # ---- bookmarks ----

    def add_bookmark(self, t: float, label: str, note: str | None = None) -> dict[str, Any]:
        """Add a bookmark at video time t (seconds). Label is the short name
        shown in the UI; note is an optional longer free-text comment."""
        return self._append(BOOKMARKS, {"t": round(float(t), 3), "label": label, "note": note})

    def list_bookmarks(self) -> list[dict[str, Any]]:
        return self._live_rows(BOOKMARKS)

    def delete_bookmark(self, annotation_id: str) -> bool:
        """Tombstone a bookmark by id. Returns True if a matching live row
        existed, False otherwise (idempotent delete — calling twice does
        nothing the second time)."""
        live = {r["annotation_id"] for r in self.list_bookmarks()}
        if annotation_id not in live:
            return False
        self._append_tombstone(BOOKMARKS, annotation_id)
        return True

    # ---- screenshots ----

    def add_screenshot(
        self,
        t: float,
        label: str,
        note: str | None = None,
        png_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        """Record a screenshot taken at video time t.

        `png_bytes` is the client-composited video frame + overlay canvas,
        uploaded as-is. The annotation layer never renders a frame — that
        compositing happens in the browser, which is the single annotated-
        frame renderer per report §2.5 (the dual-renderer hazard fix). When
        `png_bytes` is None the row is metadata-only and the user is
        expected to keep the PNG themselves; the row still anchors the
        timestamp + label + note in the review log.
        """
        row = self._append(
            SCREENSHOTS,
            {
                "t": round(float(t), 3),
                "label": label,
                "note": note,
                "png_filename": None,
            },
        )
        if png_bytes is not None:
            png_filename = f"{row['annotation_id']}.png"
            (self.screenshots_dir / png_filename).write_bytes(png_bytes)
            # Update the row in place (file rewrite is acceptable here: the
            # row was just appended in this same call, so no concurrent
            # reader can have a stale view yet, and the rewrite is within
            # the same atomic add_screenshot call).
            row["png_filename"] = png_filename
            self._rewrite_last_row(SCREENSHOTS, row)
        return row

    def _rewrite_last_row(self, kind: str, new_row: dict[str, Any]) -> None:
        """Replace the last row of <kind>.jsonl with new_row. Used by
        add_screenshot() to fill in png_filename after the PNG is written
        in the same call — the id was already minted, so we patch the row
        in place rather than appending a second one."""
        path = self._log_path(kind)
        if not path.exists():
            return
        lines = path.read_text().splitlines()
        if not lines:
            return
        lines[-1] = json.dumps(new_row, separators=(",", ":"), ensure_ascii=False)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def list_screenshots(self) -> list[dict[str, Any]]:
        return self._live_rows(SCREENSHOTS)

    def screenshot_png_path(self, annotation_id: str) -> Path | None:
        """Resolve the on-disk PNG path for a screenshot annotation, or None
        if no PNG was stored. The API uses this to serve the PNG back to the
        browser when the user revisits the run."""
        for row in self.list_screenshots():
            if row.get("annotation_id") == annotation_id:
                png = row.get("png_filename")
                if not png:
                    return None
                path = self.screenshots_dir / png
                return path if path.exists() else None
        return None

    def delete_screenshot(self, annotation_id: str) -> bool:
        """Tombstone a screenshot metadata row. The PNG file itself is left
        on disk — disk GC is not the annotation log's job, and a re-import
        or undelete could still want it."""
        live = {r["annotation_id"] for r in self.list_screenshots()}
        if annotation_id not in live:
            return False
        self._append_tombstone(SCREENSHOTS, annotation_id)
        return True

    # ---- bulk read for the UI ----

    def all_annotations(self) -> dict[str, list[dict[str, Any]]]:
        """All live annotations grouped by kind, for the UI's initial load."""
        return {kind: self._live_rows(kind) for kind in PHASE2_KINDS}

    def export_payload(self) -> dict[str, Any]:
        """Snapshot for the JSON/CSV export bundle. The export includes only
        live entries (tombstones are an internal audit detail)."""
        return {
            "bookmarks": self.list_bookmarks(),
            "screenshots": self.list_screenshots(),
        }

    def close(self) -> None:
        """No-op — file-based, nothing to close. Mirrors ArtifactStore.close()
        so callers can use the same context-manager-style pattern uniformly."""
        return
