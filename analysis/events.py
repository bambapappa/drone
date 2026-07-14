"""P5 event derivation: diff per-frame behavior/situation status into events.

Reads P2 tracklets and (if P3 ran) the tracklet→person map, replays the
carried-over BehaviorAnalyzer and SituationAnalyzer over the trajectory +
frame streams, and diffs their per-frame status output into discrete events
with onset/offset timestamps and confidence.

Phase 2 categories (per the architecture report §3 and the build-order in §7):
  STILLA     — sustained no-motion (carried-over BehaviorAnalyzer.STATUS_STILL)
  MOT_FARA   — sustained motion toward a marked/detected danger point
               (carried-over BehaviorAnalyzer.STATUS_TOWARD)
  HAZARD     — fire/smoke onset (carried-over SituationAnalyzer)

IRRATIONELL is explicitly Phase 4 per the report's build order; the
sub-signal set in §4 slots in here as another status stream feeding the same
diff, without restructuring this module.

This pass is the marriage of the report's P4 (per-frame behavior/situation
status) and P5 (status-stream diffing) into a single pass. The analyzers are
stateless per-call (their internal state is a function of the call sequence,
not the artifact), so there's no value in persisting per-frame status
separately — we compute and diff in one pass.

**Determinism.** Both analyzers are deterministic given the same call sequence
(fixed thresholds, no RNG). This pass drives them in (frame_no, tracklet_id)
order, so two runs over the same P1+P2 (+P3) output produce byte-identical
events. Mirrors the P1/P2/P3 guarantee.

**The danger point.** The live system's MOT_FARA needs an operator-marked
danger point. Offline, the natural source is the SituationAnalyzer's detected
fire/smoke position (retroactive operator-marked queries are Phase 4). If no
hazard is active in a frame, no danger point is fed to BehaviorAnalyzer that
frame and MOT_FARA cannot fire — STILLA can.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from analysis.behavior import STATUS_STILL, STATUS_TOWARD, BehaviorAnalyzer, BehaviorConfig
from analysis.orchestrator import OfflineConfig
from analysis.situation import SituationAnalyzer

# Internal category enum values stay English (per AGENTS.md convention); the
# GUI maps these to Swedish display text. A future category (IRRATIONELL,
# THREAT, ...) adds a constant here and a Swedish label in the UI registry.
CATEGORY_STILLA = "STILLA"
CATEGORY_MOT_FARA = "MOT_FARA"
CATEGORY_HAZARD = "HAZARD"

ALL_CATEGORIES = (CATEGORY_STILLA, CATEGORY_MOT_FARA, CATEGORY_HAZARD)


def _event_id(category: str, seq: int) -> str:
    """Stable, human-readable event id: <category>-<6-digit seq>.

    Determinism-friendly: the sequence is fixed because derivation processes
    frames in order and tracklets in ascending id order. Two runs over the
    same input produce identical ids."""
    return f"{category.lower()}-{seq:06d}"


@dataclass
class Event:
    """One derived event (architecture report §3 events/ schema).

    `review` is initialized by the engine to the default unreviewed state and
    never written by the engine again — review verdicts (confirm/reject/note)
    arrive through the annotations layer (Phase 3), which is a separate
    append-only log keyed to artifact version (report §2.4), never mixed into
    this AI-generated table.
    """

    event_id: str
    category: str
    person_id: int | None
    t_start: float
    t_end: float
    confidence: float
    evidence: dict[str, Any]
    review: dict[str, Any] = field(
        default_factory=lambda: {"state": "unreviewed", "note": None, "reviewer": None, "reviewed_at": None}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "category": self.category,
            "person_id": self.person_id,
            "t_start": round(self.t_start, 3),
            "t_end": round(self.t_end, 3),
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "review": self.review,
        }


def _behavior_config_from_offline(config: OfflineConfig) -> BehaviorConfig:
    return BehaviorConfig(
        window_s=config.beh_window_s,
        min_history_s=config.beh_min_history_s,
        still_speed=config.beh_still_speed,
        still_time_s=config.beh_still_time_s,
        toward_speed=config.beh_toward_speed,
        toward_angle_deg=config.beh_toward_angle_deg,
        toward_time_s=config.beh_toward_time_s,
        prone_aspect=config.beh_prone_aspect,
    )


def derive_behavior_events(
    tracklet_rows: Iterable[dict[str, Any]],
    person_by_tracklet: dict[int, int],
    fps: float,
    frame_w: int,
    frame_h: int,
    config: OfflineConfig,
    danger_px: tuple[float, float] | None = None,
) -> list[Event]:
    """Replay BehaviorAnalyzer over a tracklet table and diff into events.

    Reads P2's per-(tracklet, frame) rows (xyxy = tracker/Kalman-adjusted
    box). For each tracklet, runs BehaviorAnalyzer.update() in frame order
    with video time t = frame_no / fps, then diffs the status timeline into
    STILLA / MOT_FARA events (one event per contiguous status span ≥ the
    analyzer's required duration).

    `person_by_tracklet` maps tracklet_id → P3 person_id. When P3 ran, every
    event is tagged with its person_id; when P3 didn't run, person_id is
    None. Per the report's events/ schema, person_id is null where not
    applicable (HAZARD is null by construction; person-keyed categories are
    null only when P3 was skipped).

    `danger_px` is the danger point in pixel coordinates (frame-space),
    optional. When None, MOT_FARA cannot fire (the analyzer's
    toward_danger check requires a danger point to compute the direction
    cosine against). In the offline P5 pass, this is fed per-frame from the
    SituationAnalyzer's detected fire/smoke position; retroactive operator-
    marked queries are Phase 4.
    """
    if fps <= 0:
        return []

    # Group rows by tracklet_id, then sort each by frame_no so update() is
    # called in temporal order (the analyzer's EMA / hysteresis depends on it).
    by_tracklet: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in tracklet_rows:
        by_tracklet[int(row["tracklet_id"])].append(row)
    for tid in by_tracklet:
        by_tracklet[tid].sort(key=lambda r: int(r["frame_no"]))

    analyzer = BehaviorAnalyzer(_behavior_config_from_offline(config))

    # Per-tracklet per-frame status timeline: {tracklet_id: [(frame_no, status, prone, speed)]}
    timelines: dict[int, list[tuple[int, str, bool, float]]] = defaultdict(list)

    for tid in sorted(by_tracklet.keys()):
        for row in by_tracklet[tid]:
            frame_no = int(row["frame_no"])
            x0, y0, x1, y1 = (float(v) for v in row["xyxy"])
            # stab_pos = foot-center (the live pipeline's convention); body
            # height in pixels for body-height-normalized speed. Width/height
            # for the aspect-ratio "prone" check.
            stab_pos = ((x0 + x1) / 2.0, y1)
            box_h = max(y1 - y0, 1.0)
            aspect = max((x1 - x0) / box_h, 0.0)
            t = frame_no / fps
            status, prone, speed = analyzer.update(
                pid=tid, t=t, stab_pos=stab_pos, box_h=box_h, aspect=aspect, danger_stab=danger_px
            )
            timelines[tid].append((frame_no, status, prone, speed))
        # Free analyzer state for this tracklet so it cannot bleed into the
        # next one (BehaviorAnalyzer keeps per-pid state keyed by pid; since
        # each tid is unique that's already isolated, but we drop to keep
        # memory bounded on long films).
        analyzer.drop_inactive(set())

    # Diff each timeline into events.
    events: list[Event] = []
    seq = 0
    for tid in sorted(timelines.keys()):
        person_id = person_by_tracklet.get(tid)
        for ev in _diff_status_timeline(
            timelines[tid],
            tracklet_id=tid,
            person_id=person_id,
            fps=fps,
            seq_start=seq,
            config=config,
        ):
            events.append(ev)
            seq += 1
    return events


def _diff_status_timeline(
    timeline: list[tuple[int, str, bool, float]],
    tracklet_id: int,
    person_id: int | None,
    fps: float,
    seq_start: int,
    config: OfflineConfig,
) -> Iterable[Event]:
    """Walk a (frame_no, status, prone, speed) timeline; emit one Event per
    contiguous STILLA or MOT_FARA span.

    A status span qualifies as an event only if it lasts at least the
    analyzer's threshold for that status (still_time_s / toward_time_s) —
    mirroring what the analyzer itself required to *enter* the status in the
    first place, so an event's onset is honest about the gate that fired.
    Spans shorter than the threshold are dropped (they are jitter entering
    and leaving the status within the hysteresis window).
    """
    # Map status → (category, required_duration_s).
    category_map = {
        STATUS_STILL: (CATEGORY_STILLA, config.beh_still_time_s),
        STATUS_TOWARD: (CATEGORY_MOT_FARA, config.beh_toward_time_s),
    }

    span_status: str | None = None
    span_start_frame: int = 0
    span_start_speeds: list[float] = []
    span_prone: list[bool] = []
    seq = seq_start

    def flush(end_frame: int) -> None:
        nonlocal seq
        if span_status is None or span_status not in category_map:
            return
        # Inclusive span: end_frame is the last frame where status held, so
        # duration in frames = end - start + 1.
        duration_s = (end_frame - span_start_frame + 1) / fps
        category, threshold_s = category_map[span_status]
        # No threshold filter here: BehaviorAnalyzer already enforces
        # still_time_s / toward_time_s to ENTER the status (a span of STILL
        # means "the analyzer was confident enough to flag it"), so every
        # contiguous span is a real event. Re-filtering would drop legitimate
        # short spans after a hysteresis flip (STILL→OK→STILL within one
        # physical stillness episode) and misrepresent the analyzer's output.
        # Confidence: cap at 1.0; scale by how far past the threshold the
        # span runs (a span exactly at threshold = barely-an-event; a span
        # well past it = high confidence). This is a transparent heuristic
        # consistent with the project's "förslag" labeling (DECISIONS B6):
        # the engine never claims clinical certainty.
        confidence = min(1.0, duration_s / threshold_s / 2.0 + 0.3)
        evidence = {
            "tracklet_id": tracklet_id,
            "frame_start": span_start_frame,
            "frame_end": end_frame,
            "duration_s": round(duration_s, 3),
            "avg_speed": round(_mean(span_start_speeds), 4),
            "prone_majority": _majority(span_prone),
            "samples": len(span_start_speeds),
        }
        ev = Event(
            event_id=_event_id(category, seq),
            category=category,
            person_id=person_id,
            t_start=span_start_frame / fps,
            t_end=end_frame / fps,
            confidence=confidence,
            evidence=evidence,
        )
        events_buf.append(ev)
        seq += 1

    events_buf: list[Event] = []
    for frame_no, status, prone, speed in timeline:
        if status == span_status:
            span_start_speeds.append(speed)
            span_prone.append(prone)
            continue
        # Status change — flush the previous span (if any) up to this frame.
        if span_status is not None:
            flush(frame_no - 1)
        span_status = status if status in category_map else None
        span_start_frame = frame_no
        span_start_speeds = [speed] if status in category_map else []
        span_prone = [prone] if status in category_map else []
    if span_status is not None:
        flush(timeline[-1][0] if timeline else span_start_frame)
    return events_buf


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _majority(xs: list[bool]) -> bool:
    return sum(1 for x in xs if x) > len(xs) / 2 if xs else False


def derive_hazard_events(
    frames: Iterable,
    fps: float,
    config: OfflineConfig,
    ignore_regions: list[tuple[float, float, float, float]] | None = None,
) -> list[Event]:
    """Replay SituationAnalyzer over a frame stream and diff fire/smoke into
    HAZARD events.

    HAZARD events are not person-keyed (a fire is not a person); person_id
    is always None. `evidence.kind` distinguishes fire vs smoke. The kind is
    taken from the SituationAnalyzer's fire/smoke fields, which are gated by
    `hold_s` inside the analyzer — so by the time we see a transition into
    "fire present", it has already been sustained for hold_s, and the event
    onset is back-dated to the analyzer's `Hazard.since`.

    `ignore_regions` are the normalized PiP/IR regions to black out before
    the situation masks run (carried forward verbatim from the live tool).
    """
    if fps <= 0:
        return []

    sit = SituationAnalyzer(
        min_area=config.hazard_min_area,
        hold_s=config.hazard_hold_s,
        flow_ema=config.smoke_flow_ema,
        base_margin=config.base_margin,
        base_hysteresis=config.base_hysteresis,
        fire_require_smoke=config.fire_require_smoke,
    )

    # Per-kind timeline: fire / smoke presence per frame, plus area for
    # confidence scoring.
    fire_timeline: list[tuple[int, bool, float]] = []
    smoke_timeline: list[tuple[int, bool, float]] = []
    danger_pts: list[tuple[int, tuple[float, float] | None]] = []

    frame_no = 0
    for frame in frames:
        t = frame_no / fps
        state = sit.update(frame, t, danger_norm=None, ignore=ignore_regions)
        if state.fire is not None:
            fire_timeline.append((frame_no, True, state.fire.area))
            danger_pts.append((frame_no, state.fire.pos))
        else:
            fire_timeline.append((frame_no, False, 0.0))
            if state.smoke is not None:
                danger_pts.append((frame_no, state.smoke.pos))
            else:
                danger_pts.append((frame_no, None))
        smoke_timeline.append((frame_no, state.smoke is not None, state.smoke.area if state.smoke else 0.0))
        frame_no += 1

    events: list[Event] = []
    seq = 0
    for kind, timeline in (("fire", fire_timeline), ("smoke", smoke_timeline)):
        for ev in _diff_hazard_timeline(timeline, kind=kind, fps=fps, seq_start=seq):
            events.append(ev)
            seq += 1
    # Sort by onset so the event list reads in temporal order across kinds.
    events.sort(key=lambda e: (e.t_start, e.evidence["kind"]))
    # Re-number ids in temporal order so the event log reads naturally
    # (HAZARD-000001, HAZARD-000002, ...) regardless of which kind fired first.
    for i, ev in enumerate(events):
        ev.event_id = _event_id(CATEGORY_HAZARD, i)
    return events


def _diff_hazard_timeline(
    timeline: list[tuple[int, bool, float]],
    kind: str,
    fps: float,
    seq_start: int,
) -> Iterable[Event]:
    """Diff a (frame_no, present, area) timeline into one HAZARD event per
    contiguous present-span. The SituationAnalyzer already enforces hold_s
    internally, so any transition into present is a real onset."""
    events: list[Event] = []
    span_present = False
    span_start_frame = 0
    areas: list[float] = []
    seq = seq_start

    for frame_no, present, area in timeline:
        if present and not span_present:
            span_present = True
            span_start_frame = frame_no
            areas = [area]
        elif present and span_present:
            areas.append(area)
        elif not present and span_present:
            events.append(
                _make_hazard_event(
                    kind=kind,
                    start_frame=span_start_frame,
                    end_frame=frame_no - 1,
                    areas=areas,
                    fps=fps,
                    seq=seq,
                )
            )
            seq += 1
            span_present = False
            areas = []
    if span_present:
        events.append(
            _make_hazard_event(
                kind=kind,
                start_frame=span_start_frame,
                end_frame=timeline[-1][0] if timeline else span_start_frame,
                areas=areas,
                fps=fps,
                seq=seq,
            )
        )
    return events


def _make_hazard_event(
    kind: str,
    start_frame: int,
    end_frame: int,
    areas: list[float],
    fps: float,
    seq: int,
) -> Event:
    duration_s = (end_frame - start_frame + 1) / fps
    return Event(
        event_id=_event_id(CATEGORY_HAZARD, seq),
        category=CATEGORY_HAZARD,
        person_id=None,
        t_start=start_frame / fps,
        t_end=(end_frame + 1) / fps,
        confidence=min(1.0, _mean(areas) * 20.0 + duration_s / 4.0),
        evidence={
            "kind": kind,
            "frame_start": start_frame,
            "frame_end": end_frame,
            "duration_s": round(duration_s, 3),
            "area_mean": round(_mean(areas), 5),
            "area_peak": round(max(areas) if areas else 0.0, 5),
        },
    )


def derive_events(
    tracklet_rows: Iterable[dict[str, Any]],
    person_by_tracklet: dict[int, int],
    frames: Iterable,
    fps: float,
    frame_w: int,
    frame_h: int,
    config: OfflineConfig,
    ignore_regions: list[tuple[float, float, float, float]] | None = None,
) -> list[Event]:
    """Top-level P5 derivation: behavior + situation → events.

    Combines derive_behavior_events and derive_hazard_events into one call.
    The behavior derivation uses the *dynamic* danger point taken from the
    situation analyzer's fire/smoke detection per frame (fire preferred over
    smoke). When neither is active, danger is None that frame and MOT_FARA
    cannot fire — STILLA can.

    The two sub-derivations are independent: behavior runs over tracklets
    (cheap, no frame pixels needed beyond P2's already-adjusted boxes), and
    situation runs over the raw frame stream (cheap heuristics, no inference
    — the heavy pass is P1). P5 always re-runs in full like P2/P3, and is
    deterministic given the same P1+P2(+P3) output.

    Note on the danger point: SituationAnalyzer outputs normalized positions
    (0..1); behavior takes pixel positions. We convert per-frame so the
    spatio-temporal direction check operates in the same coordinate space as
    the tracklet boxes.
    """
    # First pass: run the situation analyzer to learn the per-frame danger
    # point (fire/smoke position). We have to materialize frames anyway
    # because SituationAnalyzer holds prev_gray internally.
    frame_list = list(frames)
    sit = SituationAnalyzer(
        min_area=config.hazard_min_area,
        hold_s=config.hazard_hold_s,
        flow_ema=config.smoke_flow_ema,
        base_margin=config.base_margin,
        base_hysteresis=config.base_hysteresis,
        fire_require_smoke=config.fire_require_smoke,
    )

    # Per-frame danger point in pixels (or None). Also the input for the
    # hazard diff, computed as a by-product so we only walk the frame stream
    # once.
    fire_timeline: list[tuple[int, bool, float]] = []
    smoke_timeline: list[tuple[int, bool, float]] = []
    danger_px_by_frame: dict[int, tuple[float, float] | None] = {}

    frame_no = 0
    for frame in frame_list:
        t = frame_no / fps
        state = sit.update(frame, t, danger_norm=None, ignore=ignore_regions)
        if state.fire is not None:
            fire_timeline.append((frame_no, True, state.fire.area))
            danger_px_by_frame[frame_no] = (state.fire.pos[0] * frame_w, state.fire.pos[1] * frame_h)
        else:
            fire_timeline.append((frame_no, False, 0.0))
            if state.smoke is not None:
                danger_px_by_frame[frame_no] = (
                    state.smoke.pos[0] * frame_w,
                    state.smoke.pos[1] * frame_h,
                )
            else:
                danger_px_by_frame[frame_no] = None
        smoke_timeline.append((frame_no, state.smoke is not None, state.smoke.area if state.smoke else 0.0))
        frame_no += 1

    # Behavior events: one derivation per contiguous run of same-danger-state
    # would be wasteful; instead, since the danger point only affects MOT_FARA
    # (STILLA is danger-independent), we run behavior with a *constant* danger
    # point equal to the time-weighted mean of all non-None danger points.
    # This is an approximation — Phase 4's retroactive hazard feature will do
    # this per-frame properly — but it captures the dominant case (one
    # sustained hazard during a film) and degrades gracefully to None when no
    # hazard ever fires.
    present_dangers = [p for p in danger_px_by_frame.values() if p is not None]
    mean_danger: tuple[float, float] | None = None
    if present_dangers:
        mean_danger = (
            sum(p[0] for p in present_dangers) / len(present_dangers),
            sum(p[1] for p in present_dangers) / len(present_dangers),
        )

    behavior_events = derive_behavior_events(
        tracklet_rows,
        person_by_tracklet=person_by_tracklet,
        fps=fps,
        frame_w=frame_w,
        frame_h=frame_h,
        config=config,
        danger_px=mean_danger,
    )

    # Hazard events: diff the timelines we already collected.
    hazard_events: list[Event] = []
    seq = 0
    for kind, timeline in (("fire", fire_timeline), ("smoke", smoke_timeline)):
        for ev in _diff_hazard_timeline(timeline, kind=kind, fps=fps, seq_start=seq):
            hazard_events.append(ev)
            seq += 1
    hazard_events.sort(key=lambda e: (e.t_start, e.evidence["kind"]))
    for i, ev in enumerate(hazard_events):
        ev.event_id = _event_id(CATEGORY_HAZARD, i)

    # Merge into a single time-ordered log. Stable sort preserves the
    # within-category ordering above.
    all_events = behavior_events + hazard_events
    all_events.sort(key=lambda e: (e.t_start, e.category, e.event_id))
    return all_events
