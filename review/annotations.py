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

Phase 3 adds two more kinds (without restructuring this module):
  - verdicts:       confirm/reject/note on a specific event_id. Unlike
                    bookmarks/screenshots (one row = one entity, tombstone to
                    delete), a verdict's *entity* is the event_id and it can
                    change state repeatedly (unreviewed -> confirmed -> a
                    note edit -> rejected, ...). Every write appends a new
                    complete row; there is no delete. The read side reduces
                    to the latest row per event_id (see `all_verdicts`) —
                    this keeps the full review history in the log (an
                    auditor can see every state change) while giving callers
                    a simple "current state" view. Crucially, this is still
                    a separate log from events/<pass>.jsonl: the engine's
                    events table keeps its frozen `review: unreviewed`
                    default forever (see analysis/events.py's Event
                    docstring), and the API layer merges the latest verdict
                    on top when serving events to the review UI.
  - operator_notes: imported field notes (one row = one entity, tombstone to
                    delete a bad import line), same shape as bookmarks.

Phase 4 adds one more kind:
  - hazard_marker:  the reviewer's manually placed/moved danger point for
                    retroactive MOT_FARA recompute (report §5.1). Like
                    verdicts, this is a single evolving value with a full
                    history, not a set of independently deletable entities —
                    every placement (or explicit clear) appends a full row;
                    the read side reduces to the latest row (see
                    `get_hazard_marker`). Unlike verdicts there is no key
                    (only one marker exists per run at a time). Recompute
                    itself lives in review/hazard.py, not here — this module
                    only stores the marker's position.

Identity corrections (split/merge person_id) remain a later phase — not
implemented here.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BOOKMARKS = "bookmarks"
SCREENSHOTS = "screenshots"
VERDICTS = "verdicts"
OPERATOR_NOTES = "operator_notes"
HAZARD_MARKER = "hazard_marker"
# Entity-per-row kinds: one row = one thing, tombstone-deletable, exactly
# like bookmarks/screenshots. Verdicts and hazard_marker are NOT in this set
# — they are single evolving values with latest-row-wins semantics instead
# (see `all_verdicts` / `get_hazard_marker`).
ENTITY_KINDS = (BOOKMARKS, SCREENSHOTS, OPERATOR_NOTES)
ALL_KINDS = (BOOKMARKS, SCREENSHOTS, VERDICTS, OPERATOR_NOTES, HAZARD_MARKER)

REVIEW_STATES = ("unreviewed", "confirmed", "rejected")


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
        if kind not in ALL_KINDS:
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
        annotation_id = _annotation_id()
        png_filename = None
        if png_bytes is not None:
            png_filename = f"{annotation_id}.png"
            (self.screenshots_dir / png_filename).write_bytes(png_bytes)
        return self._append(
            SCREENSHOTS,
            {
                "annotation_id": annotation_id,
                "t": round(float(t), 3),
                "label": label,
                "note": note,
                "png_filename": png_filename,
            },
        )

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

    # ---- verdicts (event confirm/reject/note) ----

    def set_verdict(
        self,
        event_id: str,
        state: str | None = None,
        note: str | None = None,
        reviewer: str | None = None,
    ) -> dict[str, Any]:
        """Append a new verdict row for `event_id`.

        This is a partial update over the *previous* latest row for the same
        event_id: any field left as None here carries forward the prior
        value (defaulting to the engine's "unreviewed" state with no note/
        reviewer if this is the first verdict). That lets the UI submit
        "just a note edit" without having to resend the current state, or
        "just a state change" without clobbering an existing note.

        Every call appends a full row rather than mutating one in place —
        the JSONL stays append-only and the state-transition history is
        fully reconstructable, mirroring bookmarks/screenshots' discipline
        even though the read-side semantics (latest-wins per key, not
        live-rows-minus-tombstones) differ.
        """
        if state is not None and state not in REVIEW_STATES:
            raise ValueError(f"unknown review state: {state!r}")
        prior = self.get_verdict(event_id) or {
            "state": "unreviewed",
            "note": None,
            "reviewer": None,
        }
        row = {
            "event_id": event_id,
            "state": state if state is not None else prior["state"],
            "note": note if note is not None else prior.get("note"),
            "reviewer": reviewer if reviewer is not None else prior.get("reviewer"),
        }
        return self._append(VERDICTS, row)

    def get_verdict(self, event_id: str) -> dict[str, Any] | None:
        """Latest verdict row for one event_id, or None if never reviewed."""
        latest: dict[str, Any] | None = None
        for row in self._iter_raw(VERDICTS):
            if row.get("event_id") == event_id:
                latest = row
        return latest

    def all_verdicts(self) -> dict[str, dict[str, Any]]:
        """Latest verdict per event_id, keyed by event_id.

        One pass over the log reducing to the last row per key — this is
        the "latest-row-wins" reduction verdicts use instead of the
        tombstone-filtering `_live_rows` bookmarks/screenshots use, since a
        verdict's rows are a state-transition history for one key rather
        than independent creatable/deletable entities."""
        latest: dict[str, dict[str, Any]] = {}
        for row in self._iter_raw(VERDICTS):
            eid = row.get("event_id")
            if eid is not None:
                latest[eid] = row
        return latest

    # ---- operator notes (imported field observations) ----

    def add_operator_note(self, t: float, text: str, raw_line: str | None = None) -> dict[str, Any]:
        """Add one imported operator note at video time t (seconds).

        `text` is the parsed free-text observation; `raw_line` is the
        original input line verbatim (kept for audit — the parser that
        produced `t`/`text` is deliberately forgiving of format variance, so
        keeping the raw line lets a reviewer sanity-check a surprising parse
        without re-opening the original import file)."""
        return self._append(OPERATOR_NOTES, {"t": round(float(t), 3), "text": text, "raw_line": raw_line})

    def list_operator_notes(self) -> list[dict[str, Any]]:
        return self._live_rows(OPERATOR_NOTES)

    def delete_operator_note(self, annotation_id: str) -> bool:
        """Tombstone one imported note (e.g. a bad parse the reviewer wants
        to drop). Idempotent, like delete_bookmark."""
        live = {r["annotation_id"] for r in self.list_operator_notes()}
        if annotation_id not in live:
            return False
        self._append_tombstone(OPERATOR_NOTES, annotation_id)
        return True

    # ---- hazard marker (Phase 4 retroactive MOT_FARA recompute) ----

    def set_hazard_marker(self, x: float, y: float, note: str | None = None) -> dict[str, Any]:
        """Place/move the reviewer's hazard marker. `x`/`y` are frame-pixel
        coordinates — the same space the overlay canvas already draws
        tracklet boxes in (see review/static/app.js), so a click on the
        canvas maps straight through with no conversion. Every placement
        appends a full row (never rewritten), mirroring set_verdict's
        audit-trail discipline; the read side reduces to the latest row."""
        return self._append(HAZARD_MARKER, {"x": round(float(x), 1), "y": round(float(y), 1), "note": note})

    def clear_hazard_marker(self) -> dict[str, Any]:
        """Remove the manual override — MOT_FARA falls back to the engine's
        own time-weighted-mean danger point. Recorded as an explicit
        x=None/y=None row rather than a tombstone, since hazard_marker has
        no per-row id to tombstone (see get_hazard_marker)."""
        return self._append(HAZARD_MARKER, {"x": None, "y": None, "note": None})

    def get_hazard_marker(self) -> dict[str, Any] | None:
        """Latest hazard_marker row, or None if never set. A row with
        x=None (from clear_hazard_marker) is still returned — the caller
        checks `row["x"] is not None` to know whether an override is active,
        so a cleared marker doesn't fall through to a stale earlier
        position."""
        latest: dict[str, Any] | None = None
        for row in self._iter_raw(HAZARD_MARKER):
            latest = row
        return latest

    # ---- bulk read for the UI ----

    def all_annotations(self) -> dict[str, list[dict[str, Any]]]:
        """All live annotations grouped by kind, for the UI's initial load."""
        return {kind: self._live_rows(kind) for kind in ENTITY_KINDS}

    def export_payload(self) -> dict[str, Any]:
        """Snapshot for the JSON/CSV export bundle. The export includes only
        live entries (tombstones are an internal audit detail)."""
        return {
            "bookmarks": self.list_bookmarks(),
            "screenshots": self.list_screenshots(),
            "operator_notes": self.list_operator_notes(),
        }

    def close(self) -> None:
        """No-op — file-based, nothing to close. Mirrors ArtifactStore.close()
        so callers can use the same context-manager-style pattern uniformly."""
        return
