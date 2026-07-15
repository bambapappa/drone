"""Read-time projection of manual identity corrections over P3's persons.

Phase 5 (report §3.3-4, §5.5): the reviewer can split a wrongly-merged
person or merge two wrongly-split ones from the dossier view. Two rules
govern how that works, both inherited from the phases before it:

**Corrections are annotations, not mutations.** `persons/<pass>.jsonl` is
engine output — rewritten byte-identically on every re-analysis, never
edited in place. A manual split/merge is a *human judgment about the
engine's output* — exactly the class of data the annotation log exists for
(verdicts on events, the hazard marker). It is also, per the report's own
framing, labeled ground truth for a future learned model: "person 7 and
person 12 are the same figurant" is a training label, and training labels
must survive the engine run that produced the mistake being corrected.
So each correction is one append-only op row in
`annotations/identity_corrections.jsonl`, and this module REPLAYS the live
ops over the engine's persons at read time. Undo = tombstone the op row;
the projection simply stops applying it.

**The projection is deterministic.** Ops replay in append order (the order
`AnnotationStore.list_identity_corrections` returns), new person ids from a
split are allocated as max(existing)+1 at the moment the op applies, and no
RNG or wall-clock is consulted — the same persons table + the same live ops
always produce the same projected persons, mirroring P3's own determinism
guarantee.

An op that no longer resolves (its person ids vanished because a re-analysis
renumbered P3's output, or an earlier op was undone) is *skipped and
reported*, never guessed at — the caller surfaces skipped ops so the
reviewer can see that a recorded judgment stopped applying.

Merged/split records are marked `confirmation_state: "manual"` plus
`corrected: true` so every consumer can tell human-corrected identities from
engine ones. Merged embedding centroids keep the member with the most
embedding samples rather than blending: the per-detection vectors behind the
centroids aren't at hand here, and fabricating a blended centroid would
present made-up precision as engine evidence. A split-off person gets empty
centroids for the same reason (P3 persists centroids per person, not per
tracklet).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Confirmation state for human-corrected person records. English like every
# internal enum; the UI maps it to a Swedish label ("manuellt korrigerad").
STATE_MANUAL = "manual"


@dataclass
class ProjectionResult:
    """Outcome of replaying live correction ops over the engine's persons."""

    persons: list[dict[str, Any]]
    person_by_tracklet: dict[int, int]
    applied: list[str] = field(default_factory=list)  # annotation_ids, in replay order
    skipped: list[dict[str, Any]] = field(default_factory=list)  # {annotation_id, reason}

    @property
    def changed(self) -> bool:
        return bool(self.applied)


def _span_seconds(
    tracklet_ids: list[int], tracklet_spans: dict[int, tuple[int, int]] | None, fps: float
) -> tuple[float | None, float | None]:
    """first/last_seen (video seconds) over a tracklet set, from the P2 frame
    spans — or (None, None) when spans aren't available for any member."""
    if not tracklet_spans:
        return None, None
    firsts = [tracklet_spans[t][0] for t in tracklet_ids if t in tracklet_spans]
    lasts = [tracklet_spans[t][1] for t in tracklet_ids if t in tracklet_spans]
    if not firsts or fps <= 0:
        return None, None
    return min(firsts) / fps, max(lasts) / fps


def _apply_merge(persons: dict[int, dict[str, Any]], op: dict[str, Any]) -> tuple[bool, str | None]:
    """Merge the op's person ids into the lowest one. Returns (applied, skip_reason)."""
    ids = sorted({int(p) for p in op.get("person_ids", [])})
    if len(ids) < 2:
        return False, "sammanslagning kräver minst två person-id"
    missing = [p for p in ids if p not in persons]
    if missing:
        return False, f"person(er) saknas i aktuell projektion: {missing}"
    keep_id = ids[0]
    keep = persons[keep_id]
    members = [persons[p] for p in ids]

    # Keep the centroid set of the member with the most embedding samples
    # (see module docstring: never fabricate a blended centroid).
    def sample_count(rec: dict[str, Any]) -> int:
        return sum(int(v) for v in (rec.get("embedding_counts") or {}).values())

    richest = max(members, key=lambda r: (sample_count(r), -int(r["person_id"])))

    merged = dict(keep)
    merged["tracklet_ids"] = sorted({t for m in members for t in m.get("tracklet_ids", [])})
    merged["first_seen"] = min(m.get("first_seen", 0.0) for m in members)
    merged["last_seen"] = max(m.get("last_seen", 0.0) for m in members)
    merged["embedding_centroids"] = richest.get("embedding_centroids", {})
    merged["embedding_counts"] = richest.get("embedding_counts", {})
    merged["assoc_audit"] = [a for m in members for a in (m.get("assoc_audit") or [])]
    merged["confirmation_state"] = STATE_MANUAL
    merged["corrected"] = True
    merged.setdefault("correction_ids", [])
    merged["correction_ids"] = sorted(
        set(merged["correction_ids"])
        | {cid for m in members for cid in m.get("correction_ids", [])}
        | {op["annotation_id"]}
    )

    for pid in ids[1:]:
        del persons[pid]
    persons[keep_id] = merged
    return True, None


def _apply_split(
    persons: dict[int, dict[str, Any]],
    op: dict[str, Any],
    tracklet_spans: dict[int, tuple[int, int]] | None,
    fps: float,
) -> tuple[bool, str | None]:
    """Detach the op's tracklets from its person into a new person. Returns
    (applied, skip_reason)."""
    pid = op.get("person_id")
    if pid is None or int(pid) not in persons:
        return False, f"person {pid} saknas i aktuell projektion"
    pid = int(pid)
    detach = sorted({int(t) for t in op.get("tracklet_ids", [])})
    if not detach:
        return False, "delning kräver minst ett tracklet-id"
    source = persons[pid]
    current = set(source.get("tracklet_ids", []))
    stray = [t for t in detach if t not in current]
    if stray:
        return False, f"tracklet(s) tillhör inte person {pid}: {stray}"
    if set(detach) == current:
        return False, "kan inte dela ut samtliga tracklets (blir bara ett namnbyte)"

    remaining = sorted(current - set(detach))
    new_id = max(persons) + 1

    src = dict(source)
    src["tracklet_ids"] = remaining
    fs, ls = _span_seconds(remaining, tracklet_spans, fps)
    if fs is not None:
        src["first_seen"], src["last_seen"] = round(fs, 4), round(ls, 4)
    src["confirmation_state"] = STATE_MANUAL
    src["corrected"] = True
    src["correction_ids"] = sorted(set(src.get("correction_ids", [])) | {op["annotation_id"]})
    persons[pid] = src

    fs2, ls2 = _span_seconds(detach, tracklet_spans, fps)
    persons[new_id] = {
        "person_id": new_id,
        "tracklet_ids": detach,
        # Per-tracklet centroids aren't persisted (only per-person), so a
        # split-off person honestly has no appearance centroid.
        "embedding_centroids": {},
        "embedding_counts": {},
        "first_seen": round(fs2, 4) if fs2 is not None else source.get("first_seen", 0.0),
        "last_seen": round(ls2, 4) if ls2 is not None else source.get("last_seen", 0.0),
        "confirmation_state": STATE_MANUAL,
        "assoc_audit": [],
        "corrected": True,
        "correction_ids": [op["annotation_id"]],
    }
    return True, None


def apply_corrections(
    engine_persons: list[dict[str, Any]],
    ops: list[dict[str, Any]],
    fps: float,
    tracklet_spans: dict[int, tuple[int, int]] | None = None,
) -> ProjectionResult:
    """Replay live correction ops (append order) over the engine's persons.

    `tracklet_spans` maps tracklet_id -> (first_frame, last_frame) from P2,
    used to recompute first/last_seen for split results; when absent the
    source person's original span is kept (a documented approximation, not
    an error — spans require a tracklets-table scan the caller may skip).

    Pure function: no I/O, no RNG, no wall-clock. Same inputs → same output.
    """
    persons: dict[int, dict[str, Any]] = {int(p["person_id"]): dict(p) for p in engine_persons}
    result = ProjectionResult(persons=[], person_by_tracklet={})

    for op in ops:
        kind = op.get("op")
        if kind == "merge":
            ok, reason = _apply_merge(persons, op)
        elif kind == "split":
            ok, reason = _apply_split(persons, op, tracklet_spans, fps)
        else:
            ok, reason = False, f"okänd operation: {kind!r}"
        if ok:
            result.applied.append(op["annotation_id"])
        else:
            result.skipped.append({"annotation_id": op.get("annotation_id"), "reason": reason})

    result.persons = [persons[k] for k in sorted(persons)]
    for rec in result.persons:
        for tid in rec.get("tracklet_ids", []):
            result.person_by_tracklet[int(tid)] = int(rec["person_id"])
    return result


def merge_overlap_warning(
    persons_by_id: dict[int, dict[str, Any]],
    person_ids: list[int],
    tracklet_spans: dict[int, tuple[int, int]] | None,
) -> str | None:
    """Swedish warning when a requested merge unifies persons whose tracklet
    frame spans overlap in time — the exact condition P3's hard gate treats
    as never-the-same-person. The human may know better (e.g. the tracker
    emitted twin boxes for one figurant), so the merge is allowed, but the
    contradiction is surfaced rather than silently accepted. Span
    intersection is an approximation of shared frames (a span can cover a
    frame the tracklet skipped), which errs on the side of warning."""
    if not tracklet_spans:
        return None
    spans: list[tuple[int, int, int]] = []  # (person_id, first, last)
    for pid in person_ids:
        rec = persons_by_id.get(int(pid))
        if not rec:
            continue
        member = [tracklet_spans[t] for t in rec.get("tracklet_ids", []) if t in tracklet_spans]
        if member:
            spans.append((int(pid), min(s[0] for s in member), max(s[1] for s in member)))
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a, b = spans[i], spans[j]
            if a[1] <= b[2] and b[1] <= a[2]:
                return (
                    f"Obs: person P{a[0]} och P{b[0]} syns delvis samtidigt i filmen — "
                    "två samtidiga spår är normalt olika personer. Sammanslagningen "
                    "sparas ändå, men kontrollera i videon."
                )
    return None
