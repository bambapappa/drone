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

**Ops are keyed by tracklet sets, never by person_id.** P3 assigns
person_id positionally (`enumerate(surviving, start=1)` in
analysis/identity.py), so those ids are *not* stable across re-analysis: a
renumbered run normally still has ids 1..N, meaning a person_id-keyed op
would resolve and merge the wrong pair — a guess wearing validation's
clothes. So every op records, at correction time, the FULL tracklet set of
each person it names (`member_tracklet_ids` for merge, `source_tracklet_ids`
for split), and replay resolves each named person by locating the person in
the *current projected state at that step* whose tracklet set equals the
recorded one. Exactly one match applies; anything else is skipped. The
`person_ids`/`person_id` fields stay on the row as provenance (what the
reviewer saw) and are never resolved against, never used as a fallback. The
run's `video_hash`/`config_hash` are recorded for the same reason —
diagnosis only, never a replay gate.

**Skip-and-report is the only failure mode**, and now it is honest: an op is
skipped when a recorded tracklet is absent from this run, when a recorded
tracklet set no longer matches exactly one person, when the members collapse
onto the same person, or when the op predates the stable-key schema (old
rows are never back-filled — synthesising a key from the current persons
table would invent the very evidence the key exists to check). One benign
case rides the same channel: when the engine has since associated all the
merge members' tracklets into one person, the reviewer's judgment is already
reality, and the op is reported as skipped with that reason rather than
applied. The caller surfaces skipped ops so the reviewer can see exactly why
a recorded judgment stopped applying.

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

# Reason for an op recorded before the stable tracklet-set key existed. Such
# rows carry only positional person ids, which cannot be verified against a
# re-analysed run — so they are skipped, never guessed at.
OLD_SCHEMA_REASON = "korrigering saknar tracklet-nyckel (gammalt schema)"


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


def _tracklet_set(raw: Any) -> set[int] | None:
    """Normalize a recorded tracklet-set key, or None when the op carries no
    usable key (missing field, empty list, non-numeric contents)."""
    if not isinstance(raw, (list, tuple, set)) or not raw:
        return None
    try:
        return {int(t) for t in raw}
    except (TypeError, ValueError):
        return None


def _person_tracklets(rec: dict[str, Any]) -> set[int]:
    return {int(t) for t in rec.get("tracklet_ids", [])}


def _resolve_by_tracklets(
    persons: dict[int, dict[str, Any]], want: set[int]
) -> tuple[int | None, str | None]:
    """Locate the person in the CURRENT projected state whose tracklet set
    equals `want`. Returns (person_id, None) on an unambiguous match, else
    (None, Swedish reason) — the op's own person_id is never consulted."""
    known = {t for rec in persons.values() for t in _person_tracklets(rec)}
    absent = sorted(want - known)
    if absent:
        return None, f"spår saknas i den här körningen: {absent}"
    matches = [pid for pid, rec in persons.items() if _person_tracklets(rec) == want]
    if len(matches) != 1:
        return None, (
            f"spåruppsättningen {sorted(want)} motsvarar inte längre exakt en person i den här körningen"
        )
    return matches[0], None


def _apply_merge(persons: dict[int, dict[str, Any]], op: dict[str, Any]) -> tuple[bool, str | None]:
    """Merge the op's members (resolved by their recorded tracklet sets) into
    the lowest surviving person id. Returns (applied, skip_reason)."""
    raw_keys = op.get("member_tracklet_ids")
    if not isinstance(raw_keys, list) or len(raw_keys) < 2:
        return False, OLD_SCHEMA_REASON
    wants = [_tracklet_set(k) for k in raw_keys]
    if any(w is None for w in wants):
        return False, OLD_SCHEMA_REASON
    if len({frozenset(w) for w in wants}) < 2:
        return False, "sammanslagning kräver minst två olika personer"

    resolved: list[int] = []
    failure: str | None = None
    for want in wants:
        pid, reason = _resolve_by_tracklets(persons, want)
        if pid is None:
            failure = failure or reason
        else:
            resolved.append(pid)

    if failure is not None or len(set(resolved)) != len(wants):
        # The engine may simply have caught up with the reviewer: one person
        # now holds exactly the union of the members' tracklets. That is the
        # judgment already being true, reported honestly rather than applied.
        union = set().union(*wants)
        if sum(1 for rec in persons.values() if _person_tracklets(rec) == union) == 1:
            return False, "motorn associerar redan dessa tracklets som en person"
        return False, failure or "sammanslagningen pekar på samma person mer än en gång"

    ids = sorted(resolved)
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
    """Detach the op's tracklets from its source person (resolved by the
    source's recorded tracklet set) into a new person. Returns (applied,
    skip_reason)."""
    want = _tracklet_set(op.get("source_tracklet_ids"))
    if want is None:
        return False, OLD_SCHEMA_REASON
    pid, reason = _resolve_by_tracklets(persons, want)
    if pid is None:
        return False, reason
    detach = sorted({int(t) for t in op.get("tracklet_ids", [])})
    if not detach:
        return False, "delning kräver minst ett tracklet-id"
    source = persons[pid]
    current = _person_tracklets(source)
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

    Each op names its persons by the tracklet set recorded at correction
    time, and resolution happens against the state projected SO FAR (engine
    persons plus every earlier applied op) — not against the engine table,
    since a split allocates ids that have no engine meaning. Unresolvable
    ops land in `skipped` with a Swedish reason; nothing is ever guessed.

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
