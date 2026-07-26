"""Ground-truth ("facit") scoring: reference truth vs AI events vs operator.

Phase 5 (report §5.6): the exercise leader knows the script — "3 figuranter:
1 ligger vid bilen från 0:30, …". Entered as reference truth, the tool
scores BOTH the AI's event log and (when Phase 3 imported them) the
operator's field notes against it, turning the tool from "AI vs operator"
into a training instrument for both sides.

**Format: the operator-notes format, verbatim.** A reference entry is a
timed observation exactly like an operator note ("0:30, figurant 1 lägger
sig vid bilen"), so the import reuses `review/operator_notes.py`'s forgiving
parser — same accepted time formats, same delimiter tolerance, same
warnings-not-failures behavior. Untimed free text can't be scored (there is
nothing to align it against; see below) and becomes an import warning, not a
silent drop. One format for both kinds of human-timed input means one
parser to maintain and one syntax for the exercise leader to learn.

**Scoring is time-proximity only**, for the same reason the Phase 3
comparison is: reference text and AI event categories share no vocabulary
the tool could reliably cross-match — matching on content would be
guessing. Two independent greedy nearest-time assignments (the shared
`greedy_time_match`, one-to-one, deterministic): reference vs AI events,
and reference vs operator notes. Per reference entry the result says
"hittad av AI (Δt)" / "hittad av operatör (Δt)" / missed by either, plus
the leftovers on each side (AI events and operator notes matching no
reference entry — possible false positives, or simply things the script
didn't cover; the human decides which).

**Nothing here is persisted.** Like the Phase 3 comparison, the score is
recomputed from the live reference entries + live events + live operator
notes on every call — reproducible by construction, and always consistent
with the current review state (verdict overlays, hazard-marker recompute,
identity corrections all flow in through the events the caller passes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from review.comparison import DEFAULT_TOLERANCE_S, greedy_time_match


@dataclass(frozen=True)
class GroundTruthScore:
    tolerance_s: float
    # One row per reference entry: {"gt": row, "ai": {"event", "delta_s"}|None,
    # "operator": {"note", "delta_s"}|None}. delta_s > 0 = found AFTER the
    # reference time (late detection); < 0 = before.
    entries: list[dict[str, Any]]
    ai_unmatched: list[dict[str, Any]]  # events matching no reference entry
    operator_unmatched: list[dict[str, Any]]  # notes matching no reference entry

    @property
    def counts(self) -> dict[str, int]:
        ai_found = sum(1 for e in self.entries if e["ai"] is not None)
        op_found = sum(1 for e in self.entries if e["operator"] is not None)
        return {
            "gt_total": len(self.entries),
            "ai_found": ai_found,
            "ai_missed": len(self.entries) - ai_found,
            "operator_found": op_found,
            "operator_missed": len(self.entries) - op_found,
            "ai_unmatched": len(self.ai_unmatched),
            "operator_unmatched": len(self.operator_unmatched),
        }


def score_ground_truth(
    gt_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    operator_notes: list[dict[str, Any]],
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> GroundTruthScore:
    """Score AI events and operator notes against the reference truth.

    `gt_rows` and `operator_notes` are annotation rows (need `annotation_id`,
    `t`); `events` are served event records (need `event_id`, `t_start`).
    Two independent one-to-one matchings — an AI event and an operator note
    never compete for the same reference entry.
    """
    ai_triples = greedy_time_match(
        gt_rows,
        events,
        t_a="t",
        t_b="t_start",
        id_a="annotation_id",
        id_b="event_id",
        tolerance_s=tolerance_s,
    )
    op_triples = greedy_time_match(
        gt_rows,
        operator_notes,
        t_a="t",
        t_b="t",
        id_a="annotation_id",
        id_b="annotation_id",
        tolerance_s=tolerance_s,
    )
    ai_by_gt = {gt["annotation_id"]: (ev, delta) for gt, ev, delta in ai_triples}
    op_by_gt = {gt["annotation_id"]: (note, delta) for gt, note, delta in op_triples}

    entries = []
    for gt in sorted(gt_rows, key=lambda r: (r["t"], r["annotation_id"])):
        ai = ai_by_gt.get(gt["annotation_id"])
        op = op_by_gt.get(gt["annotation_id"])
        entries.append(
            {
                "gt": gt,
                "ai": {"event": ai[0], "delta_s": round(ai[1], 3)} if ai else None,
                "operator": {"note": op[0], "delta_s": round(op[1], 3)} if op else None,
            }
        )

    matched_events = {ev["event_id"] for _, ev, _ in ai_triples}
    matched_notes = {n["annotation_id"] for _, n, _ in op_triples}
    ai_unmatched = sorted(
        (e for e in events if e["event_id"] not in matched_events), key=lambda e: e["t_start"]
    )
    operator_unmatched = sorted(
        (n for n in operator_notes if n["annotation_id"] not in matched_notes), key=lambda n: n["t"]
    )
    return GroundTruthScore(
        tolerance_s=tolerance_s,
        entries=entries,
        ai_unmatched=ai_unmatched,
        operator_unmatched=operator_unmatched,
    )
