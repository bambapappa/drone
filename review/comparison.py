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


def greedy_time_match(
    items_a: list[dict[str, Any]],
    items_b: list[dict[str, Any]],
    t_a: str,
    t_b: str,
    id_a: str,
    id_b: str,
    tolerance_s: float,
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    """Deterministic greedy nearest-time one-to-one assignment between two
    lists of timed records. Returns (a, b, delta_s) triples with
    delta_s = b[t_b] - a[t_a].

    Shared by the AI-vs-operator comparison (Phase 3) and ground-truth
    scoring (Phase 5) so the two can never drift on matching semantics.
    All within-tolerance pairs are sorted by (|delta|, a's id, b's time,
    b's id) and claimed smallest-delta first — the same inputs always
    produce the same match regardless of input list order (the project's
    "no RNG, fixed evaluation order" determinism convention).
    """
    candidates: list[tuple[float, Any, float, Any, dict[str, Any], dict[str, Any]]] = []
    for a in items_a:
        for b in items_b:
            delta = b[t_b] - a[t_a]
            if abs(delta) <= tolerance_s:
                candidates.append((abs(delta), a[id_a], b[t_b], b[id_b], a, b))
    candidates.sort(key=lambda c: c[:4])

    matched_a: set[Any] = set()
    matched_b: set[Any] = set()
    out: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for _, _, _, _, a, b in candidates:
        if a[id_a] in matched_a or b[id_b] in matched_b:
            continue
        matched_a.add(a[id_a])
        matched_b.add(b[id_b])
        out.append((a, b, b[t_b] - a[t_a]))
    return out


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
    # Tie-break order (|delta|, event_id, note time, note annotation_id) —
    # independent of the order `events`/`notes` were passed in; see
    # greedy_time_match.
    triples = greedy_time_match(
        events, notes, t_a="t_start", t_b="t", id_a="event_id", id_b="annotation_id", tolerance_s=tolerance_s
    )
    matched_events = {ev["event_id"] for ev, _, _ in triples}
    matched_notes = {note["annotation_id"] for _, note, _ in triples}
    both = [Match(event=ev, note=note, delta_s=delta) for ev, note, delta in triples]

    both.sort(key=lambda m: m.note["t"])
    ai_only = sorted((e for e in events if e["event_id"] not in matched_events), key=lambda e: e["t_start"])
    operator_only = sorted(
        (n for n in notes if n["annotation_id"] not in matched_notes), key=lambda n: n["t"]
    )

    return ComparisonResult(tolerance_s=tolerance_s, both=both, ai_only=ai_only, operator_only=operator_only)
