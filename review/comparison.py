"""Compare AI-derived events against imported operator field notes.

Report §2.4/§5.3 requirement: for each exercise, report three buckets —
found by both, AI-only, operator-only — with a time-to-detection delta
where a note and an event align. This is a pure, on-demand computation over
two already-persisted sources (P5 events + imported operator_notes
annotations): nothing here is itself persisted, so re-running it is always
reproducible given the same events + the same imported notes (no RNG, no
wall-clock lookups, no external state).

**Matching is time-only, not content-based.** An operator's note ("2
personer vid fordonet") and an AI event carry no shared vocabulary the tool
could reliably cross-reference (the note is free text; the event is a
category + evidence dict) — attempting NLP-style content matching would be
guessing, not comparing. Time proximity is the one dimension both sides
share unambiguously, and it is what the report asks for.

**Matching algorithm: greedy nearest-time, one-to-one.** Both sides are
typically small (tens of events/notes per exercise, not thousands), so a
simple deterministic greedy assignment — sort all within-tolerance
(event, note) pairs by |delta|, then assign the smallest-delta pair first,
skipping anything already claimed — is sufficient and easy to audit. Ties
break on event_id then note time then note annotation_id, so the same
inputs always produce the same match regardless of input list order
(mirrors the project's "no RNG, fixed evaluation order" determinism
convention used by P3 identity association).

**Default tolerance: 60 seconds.** Operator field notes in the example
format ("... 14:32") are minute-granular — there is no reason to expect
sub-minute precision from someone jotting down a sighting while running an
exercise. 60s absorbs that rounding plus a plausible few-seconds delay
between an observation and writing it down, without being so wide that
unrelated events across the film start colliding. It is a heuristic, not a
protocol constant, so callers may override it per exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_TOLERANCE_S = 60.0


@dataclass(frozen=True)
class Match:
    event: dict[str, Any]
    note: dict[str, Any]
    delta_s: float  # note.t - event.t_start; positive = AI detected first


@dataclass(frozen=True)
class ComparisonResult:
    tolerance_s: float
    both: list[Match]
    ai_only: list[dict[str, Any]]
    operator_only: list[dict[str, Any]]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "both": len(self.both),
            "ai_only": len(self.ai_only),
            "operator_only": len(self.operator_only),
        }


def compare_events_to_notes(
    events: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> ComparisonResult:
    """Align AI events to operator notes by time proximity.

    `events` are P5 event records (need `event_id`, `t_start`); `notes` are
    live operator_notes annotation rows (need `annotation_id`, `t`). Returns
    matched pairs plus the unmatched remainder on each side.
    """
    candidates: list[tuple[float, str, float, str, dict[str, Any], dict[str, Any]]] = []
    for ev in events:
        for note in notes:
            delta = ev["t_start"] - note["t"]
            if abs(delta) <= tolerance_s:
                candidates.append((abs(delta), ev["event_id"], note["t"], note["annotation_id"], ev, note))
    # Deterministic tie-break: |delta|, then event_id, then note time, then
    # note annotation_id — independent of the order `events`/`notes` were
    # passed in.
    candidates.sort(key=lambda c: c[:4])

    matched_events: set[str] = set()
    matched_notes: set[str] = set()
    both: list[Match] = []
    for _, _, _, _, ev, note in candidates:
        eid = ev["event_id"]
        nid = note["annotation_id"]
        if eid in matched_events or nid in matched_notes:
            continue
        matched_events.add(eid)
        matched_notes.add(nid)
        both.append(Match(event=ev, note=note, delta_s=note["t"] - ev["t_start"]))

    both.sort(key=lambda m: m.note["t"])
    ai_only = sorted((e for e in events if e["event_id"] not in matched_events), key=lambda e: e["t_start"])
    operator_only = sorted(
        (n for n in notes if n["annotation_id"] not in matched_notes), key=lambda n: n["t"]
    )

    return ComparisonResult(tolerance_s=tolerance_s, both=both, ai_only=ai_only, operator_only=operator_only)
