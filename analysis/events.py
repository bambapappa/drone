"""P5 event derivation: diff per-frame behavior/situation status into events.

Reads P2 tracklets and (if P3 ran) the tracklet→person map, replays the
carried-over BehaviorAnalyzer and SituationAnalyzer over the trajectory +
frame streams, and diffs their per-frame status output into discrete events
with onset/offset timestamps and confidence.

Categories (per the architecture report §3 and the build-order in §7):
  STILLA       — sustained no-motion (carried-over BehaviorAnalyzer.STATUS_STILL)
  MOT_FARA     — sustained motion toward a marked/detected danger point
                 (carried-over BehaviorAnalyzer.STATUS_TOWARD)
  IRRATIONELL  — Phase 4 heuristic ensemble over the same trajectories
                 (analysis/irrational.py); STILLA takes precedence over it
                 (see derive_events), MOT_FARA has no precedence interaction
                 with either.
  HAZARD       — internal compatibility enum for a persistent BRAND incident;
                 fire/smoke are observations attached to that incident

This pass is the marriage of the report's P4 (per-frame behavior/situation
status) and P5 (status-stream diffing) into a single pass. The analyzers are
stateless per-call (their internal state is a function of the call sequence,
not the artifact), so there's no value in persisting per-frame status
separately — we compute and diff in one pass.

**Determinism.** Both analyzers are deterministic given the same call sequence
(fixed thresholds, no RNG). This pass drives them in (frame_no, tracklet_id)
order, so two runs over the same P1+P2 (+P3) output produce byte-identical
events. Mirrors the P1/P2/P3 guarantee.

**Danger targets.** Offline, every active place-bound BRAND incident is a
danger target. BehaviorAnalyzer evaluates movement against the full set and
MOT_FARA evidence records the matching brand_id. A detected incident persists
through visual detector gaps until video end, but local scene identities are
never guessed across an unlinked B29 segment. Retroactive operator-marked
queries remain the Phase 4 single-point human override.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from analysis.behavior import STATUS_STILL, STATUS_TOWARD, BehaviorAnalyzer, BehaviorConfig
from analysis.brand import BrandIncident, BrandIncidentTracker, BrandObservation
from analysis.irrational import CATEGORY_IRRATIONELL, IrrationalConfig, derive_irrational_events
from analysis.orchestrator import OfflineConfig
from analysis.situation import WORK_W, SituationAnalyzer

# Internal category enum values stay English (per AGENTS.md convention); the
# GUI maps these to Swedish display text. A future category (THREAT, ...)
# adds a constant here and a Swedish label in the UI registry.
CATEGORY_STILLA = "STILLA"
CATEGORY_MOT_FARA = "MOT_FARA"
CATEGORY_HAZARD = "HAZARD"

ALL_CATEGORIES = (CATEGORY_STILLA, CATEGORY_MOT_FARA, CATEGORY_IRRATIONELL, CATEGORY_HAZARD)

DangerSet = tuple[float, float] | Mapping[str, tuple[float, float]] | None
DangerResolver = Callable[[int, int | None], DangerSet]


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


def _irrational_config_from_offline(config: OfflineConfig) -> IrrationalConfig:
    return IrrationalConfig(
        erratic_window_s=config.irr_erratic_window_s,
        erratic_tortuosity=config.irr_erratic_tortuosity,
        erratic_min_speed_bh=config.irr_erratic_min_speed_bh,
        erratic_heading_circ_var=config.irr_erratic_heading_circ_var,
        sprint_speed_bh=config.irr_sprint_speed_bh,
        sprint_time_s=config.irr_sprint_time_s,
        sprint_group_multiple=config.irr_sprint_group_multiple,
        counterflow_angle_deg=config.irr_counterflow_angle_deg,
        counterflow_time_s=config.irr_counterflow_time_s,
        counterflow_min_neighbors=config.irr_counterflow_min_neighbors,
        counterflow_radius_bh=config.irr_counterflow_radius_bh,
        counterflow_coherence=config.irr_counterflow_coherence,
        oscillation_window_s=config.irr_oscillation_window_s,
        oscillation_min_reversals=config.irr_oscillation_min_reversals,
        oscillation_min_excursion_bh=config.irr_oscillation_min_excursion_bh,
        freeze_speed_bh=config.irr_freeze_speed_bh,
        freeze_time_s=config.irr_freeze_time_s,
        bolt_speed_bh=config.irr_bolt_speed_bh,
        bolt_within_s=config.irr_bolt_within_s,
        freeze_bolt_hold_s=config.irr_freeze_bolt_hold_s,
        weight_erratic=config.irr_weight_erratic,
        weight_sprint=config.irr_weight_sprint,
        weight_counterflow=config.irr_weight_counterflow,
        weight_oscillation=config.irr_weight_oscillation,
        weight_freeze_bolt=config.irr_weight_freeze_bolt,
        score_threshold=config.irr_score_threshold,
        sustain_s=config.irr_sustain_s,
    )


def derive_behavior_events(
    tracklet_rows: Iterable[dict[str, Any]],
    person_by_tracklet: dict[int, int],
    fps: float,
    frame_w: int,
    frame_h: int,
    config: OfflineConfig,
    danger_for_frame: DangerResolver | None = None,
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

    `danger_for_frame(frame_no, segment)` returns either one legacy point or
    `{brand_id: point}` for every active incident in the analyzer's coordinate
    space. When absent/empty, MOT_FARA cannot fire that frame — STILLA can.
    The review layer's human marker remains a compatible scalar override.
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

    # Per-tracklet per-frame status timeline. brand_id identifies which one of
    # several simultaneous fire sites caused MOT_FARA; legacy/manual scalar
    # danger points keep it None.
    timelines: dict[int, list[tuple[int, str, bool, float, str | None]]] = defaultdict(list)

    for tid in sorted(by_tracklet.keys()):
        previous_segment: int | None = None
        for row in by_tracklet[tid]:
            frame_no = int(row["frame_no"])
            x0, y0, x1, y1 = (float(v) for v in row["xyxy"])
            raw_box_h = max(y1 - y0, 1.0)
            # stab_pos = foot-center (the live pipeline's convention); body
            # height in pixels for body-height-normalized speed. Width/height
            # for the aspect-ratio "prone" check.
            segment = int(row["scene_segment"]) if row.get("scene_segment") is not None else None
            if row.get("scene_pos") is not None:
                stab_pos = tuple(float(v) for v in row["scene_pos"])
                box_h = max(float(row.get("scene_box_h", raw_box_h)), 1.0)
            else:
                stab_pos = ((x0 + x1) / 2.0, y1)
                box_h = raw_box_h
            if previous_segment is not None and segment is not None and segment != previous_segment:
                # B29 local maps are unrelated across a visual loss. Reset the
                # motion history so the coordinate jump cannot become a sprint,
                # MOT_FARA, or other fabricated behavior.
                analyzer.drop_inactive(set())
            previous_segment = segment
            # Prone posture is an image-box shape measurement.  Keep both
            # width and height in raw frame pixels; scene_box_h is only for
            # motion normalization and may include camera scale/rotation.
            aspect = max((x1 - x0) / raw_box_h, 0.0)
            t = frame_no / fps
            danger = danger_for_frame(frame_no, segment) if danger_for_frame is not None else None
            status, prone, speed = analyzer.update(
                pid=tid, t=t, stab_pos=stab_pos, box_h=box_h, aspect=aspect, danger_stab=danger
            )
            brand_id = analyzer.toward_danger_of(tid) if status == STATUS_TOWARD else None
            timelines[tid].append((frame_no, status, prone, speed, brand_id))
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
    timeline: list[tuple[int, str, bool, float, str | None]],
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
    span_brand_id: str | None = None
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
        if category == CATEGORY_MOT_FARA and span_brand_id is not None:
            evidence["brand_id"] = span_brand_id
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
    for frame_no, status, prone, speed, brand_id in timeline:
        current_brand_id = brand_id if status == STATUS_TOWARD else None
        if status == span_status and current_brand_id == span_brand_id:
            span_start_speeds.append(speed)
            span_prone.append(prone)
            continue
        # Status change — flush the previous span (if any) up to this frame.
        if span_status is not None:
            flush(frame_no - 1)
        span_status = status if status in category_map else None
        span_brand_id = current_brand_id
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


def build_danger_resolver(
    danger_px_by_frame: dict[int, DangerSet],
    scene_frames: dict[int, dict[str, Any]] | None,
) -> DangerResolver | None:
    """Build a per-frame danger resolver in the BehaviorAnalyzer's space.

    For each frame with one or several danger points: if that frame has its
    own `frame_to_scene` matrix (B29), transform every point into that scene;
    otherwise keep pixel coordinates. Multiple brand ids are never averaged
    or collapsed. Frames without danger resolve to None.

    Returns None when no frame ever had a danger point, so the caller can omit
    MOT_FARA entirely. This replaces the old mean-danger collapse, which
    averaged every fire/smoke position into one constant — a fabricated point
    measured on no frame that landed in empty space when fire moved (a breach
    of B29's own anti-fabrication principle). Pure and deterministic.
    """
    resolved: dict[int, DangerSet] = {}
    resolved_segments: dict[int, int | None] = {}
    for frame_no, danger in danger_px_by_frame.items():
        if danger is None:
            continue
        scene_rec = (scene_frames or {}).get(frame_no)
        matrix = scene_rec.get("frame_to_scene") if scene_rec is not None else None
        resolved_segments[frame_no] = (
            int(scene_rec["scene_segment"])
            if matrix is not None and scene_rec.get("scene_segment") is not None
            else None
        )
        if isinstance(danger, Mapping):
            from analysis.scene import transform_point

            resolved[frame_no] = {
                brand_id: transform_point(matrix, point) if matrix is not None else point
                for brand_id, point in danger.items()
            }
        else:
            if matrix is not None:
                from analysis.scene import transform_point

                resolved[frame_no] = transform_point(matrix, danger)
            else:
                resolved[frame_no] = danger
    if not resolved:
        return None

    def resolver(frame_no: int, segment: int | None) -> DangerSet:
        expected_segment = resolved_segments.get(frame_no)
        if expected_segment is not None and segment != expected_segment:
            return None
        return resolved.get(frame_no)

    return resolver


def derive_hazard_events(
    frames: Iterable,
    fps: float,
    config: OfflineConfig,
    ignore_regions: list[tuple[float, float, float, float]] | None = None,
) -> list[Event]:
    """Replay SituationAnalyzer and aggregate its signals into BRAND events.

    HAZARD remains the sidecar enum for compatibility; evidence.kind is
    `brand`, while evidence.signals audits whether smoke, fire-colour, or both
    were observed. Every place-bound incident remains active to video end.

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
        smoke_window_s=config.hazard_smoke_window_s,
        smoke_texture_min=config.hazard_texture_min,
    )

    frame_no = 0
    tracker: BrandIncidentTracker | None = None
    for frame in frames:
        if tracker is None:
            frame_h, frame_w = frame.shape[:2]
            tracker = BrandIncidentTracker(association_radius=sit.fire_smoke_radius * max(frame_w, frame_h))
        t = frame_no / fps
        state = sit.update(frame, t, danger_norm=None, ignore=ignore_regions)
        observations = []
        for hazard in (*state.fire_hazards, *state.smoke_hazards):
            if hazard.observed:
                observations.append(
                    BrandObservation(
                        frame_no=frame_no,
                        signal=hazard.kind,
                        pos=(hazard.pos[0] * frame_w, hazard.pos[1] * frame_h),
                        area=hazard.area,
                        scene_segment=None,
                        frame_pos=(hazard.pos[0] * frame_w, hazard.pos[1] * frame_h),
                    )
                )
        tracker.observe(observations)
        frame_no += 1

    if tracker is None or frame_no == 0:
        return []
    return _brand_incidents_to_events(tracker.incidents(end_frame=frame_no - 1, fps=fps), fps)


def _brand_incidents_to_events(incidents: list[BrandIncident], fps: float) -> list[Event]:
    """Serialize persistent fire sites as the existing HAZARD category.

    `kind=brand` is the operational meaning.  `signals` preserves whether the
    detector saw smoke, fire-colour, or both without presenting those signals
    as separate incidents.
    """
    events = []
    for seq, incident in enumerate(sorted(incidents, key=lambda item: item.brand_id)):
        observed_duration_s = incident.observed_frame_count / fps
        events.append(
            Event(
                event_id=_event_id(CATEGORY_HAZARD, seq),
                category=CATEGORY_HAZARD,
                person_id=None,
                t_start=incident.t_start,
                t_end=incident.t_end,
                confidence=min(1.0, incident.area_mean * 20.0 + observed_duration_s / 4.0),
                evidence={
                    "kind": "brand",
                    "brand_id": incident.brand_id,
                    "signals": sorted(incident.signals),
                    "scene_segment": incident.scene_segment,
                    "anchor": [round(value, 3) for value in incident.anchor],
                    # The scene anchor is for association/audit.  Replaying
                    # it as one fixed screen point is wrong for a circling
                    # camera, so retain the observed frame-pixel positions.
                    "position_space": "frame_pixels",
                    "position_samples": [
                        [frame, round(x, 3), round(y, 3)] for frame, x, y in incident.position_samples
                    ],
                    "frame_start": incident.frame_start,
                    "frame_end": incident.frame_end,
                    "last_observed_frame": incident.last_observed_frame,
                    "observation_count": incident.observation_count,
                    "observed_frame_count": incident.observed_frame_count,
                    "duration_s": round(incident.t_end - incident.t_start, 3),
                    "area_mean": round(incident.area_mean, 5),
                    "area_peak": round(incident.area_peak, 5),
                    "status": "active_at_video_end",
                },
            )
        )
    return events


def _diff_and_number_hazard_events(
    fire_timeline: list[tuple[int, bool, float]],
    smoke_timeline: list[tuple[int, bool, float]],
    fps: float,
) -> list[Event]:
    """Diff fire/smoke timelines into HAZARD events, ordered and re-numbered.

    Shared by derive_hazard_events and derive_events so the two callers can't
    drift on the diff/sort/renumber logic.
    """
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
    scene_frames: dict[int, dict[str, Any]] | None = None,
) -> list[Event]:
    """Top-level P5 derivation: behavior + situation → events.

    Combines derive_behavior_events and derive_hazard_events into one call.
    The behavior derivation uses every active place-bound BRAND incident as a
    dynamic danger target. When none exists, MOT_FARA cannot fire — STILLA can.

    The two sub-derivations are independent: behavior runs over tracklets
    (cheap, no frame pixels needed beyond P2's already-adjusted boxes), and
    situation runs over the raw frame stream (cheap heuristics, no inference
    — the heavy pass is P1). P5 always re-runs in full like P2/P3, and is
    deterministic given the same P1+P2(+P3) output.

    Note on the danger point: SituationAnalyzer outputs normalized positions
    (0..1); behavior takes pixel positions. We convert per-frame so the
    spatio-temporal direction check operates in the same coordinate space as
    the tracklet boxes.

    Smoke is measured in scene space: when consecutive frames are linked by
    P2's persisted scene transforms (same segment, both records carrying
    matrices), the prev→cur affine is passed to the analyzer so the camera's
    own motion is warped out before the frame diff; an unlinked pair (segment
    break / missing transform) yields an empty motion mask that frame — never
    a guess across a visual loss.
    """
    # First pass: run the situation analyzer to learn the per-frame danger
    # point (fire/smoke position). SituationAnalyzer only needs frames in
    # sequential order (it holds prev_gray internally) — the stream is
    # consumed once, not materialized, so this stays bounded on long films.
    sit = SituationAnalyzer(
        min_area=config.hazard_min_area,
        hold_s=config.hazard_hold_s,
        flow_ema=config.smoke_flow_ema,
        base_margin=config.base_margin,
        base_hysteresis=config.base_hysteresis,
        fire_require_smoke=config.fire_require_smoke,
        smoke_window_s=config.hazard_smoke_window_s,
        smoke_texture_min=config.hazard_texture_min,
    )

    # B32: the smoke reference spans the smoke window, not one frame —
    # real smoke drifts ~1-2 px/frame and a k=1 diff fragments below the
    # absdiff threshold. Minimum span is 2 frames.
    k = max(2, round(config.hazard_smoke_window_s * fps))

    # Persistent, place-bound BRAND incidents.  Fire/smoke remain detector
    # signals; the operational incident and MOT_FARA substrate are shared.
    brand_tracker = BrandIncidentTracker(association_radius=sit.fire_smoke_radius * max(frame_w, frame_h))
    dangers_by_frame: dict[int, dict[str, tuple[float, float]]] = {}
    danger_segments_by_frame: dict[int, int | None] = {}

    frame_no = 0
    scene_hist: dict[int, dict[str, Any]] = {}  # last k+1 scene records
    # The smoke diff runs on the analyzer's downscaled frame (WORK_W wide),
    # so the prev→cur affine — in full-resolution frame pixels — is scaled to
    # that space before it reaches cv2.warpAffine.
    warp_scale = WORK_W / float(frame_w) if frame_w > 0 else 1.0
    for frame in frames:
        t = frame_no / fps
        scene_rec = (scene_frames or {}).get(frame_no)
        prev_to_cur = None
        ref_rec = scene_hist.get(frame_no - k)
        if (
            scene_rec is not None
            and scene_rec.get("frame_to_scene") is not None
            and scene_rec.get("scene_to_frame") is not None
            and ref_rec is not None
            and ref_rec.get("frame_to_scene") is not None
            and int(scene_rec.get("scene_segment", -1)) == int(ref_rec.get("scene_segment", -2))
        ):
            # Composite over the k-frame span: scene_to_frame(n) @
            # frame_to_scene(n-k) — only the translation column scales under
            # S·A·S⁻¹ (pattern pinned by test_rotation_scaling...).
            rel = np.asarray(scene_rec["scene_to_frame"], dtype=np.float32) @ np.asarray(
                ref_rec["frame_to_scene"], dtype=np.float32
            )
            rel23 = np.asarray(rel, dtype=np.float32)[:2, :3]
            # S·A·S⁻¹ for uniform scale s = warp_scale: translation scales by
            # s, the linear part is invariant. (situation.py resizes by width
            # with preserved aspect, so both axes scale by the same s.)
            rel23[:, 2] *= warp_scale
            prev_to_cur = rel23
        state = sit.update(
            frame,
            t,
            danger_norm=None,
            ignore=ignore_regions,
            prev_to_cur=prev_to_cur,
            scene_motion=True,
            ref_lag=k,
        )
        if scene_rec is not None:
            scene_hist[frame_no] = scene_rec
            scene_hist.pop(frame_no - k, None)
        coordinate_segment = (
            int(scene_rec["scene_segment"])
            if scene_rec is not None and scene_rec.get("frame_to_scene") is not None
            else None
        )
        observations = []
        for hazard in (*state.fire_hazards, *state.smoke_hazards):
            if not hazard.observed:
                continue
            pos = (hazard.pos[0] * frame_w, hazard.pos[1] * frame_h)
            if scene_rec is not None and scene_rec.get("frame_to_scene") is not None:
                from analysis.scene import transform_point

                pos = transform_point(scene_rec["frame_to_scene"], pos)
            observations.append(
                BrandObservation(
                    frame_no=frame_no,
                    signal=hazard.kind,
                    pos=pos,
                    area=hazard.area,
                    scene_segment=coordinate_segment,
                    frame_pos=(hazard.pos[0] * frame_w, hazard.pos[1] * frame_h),
                )
            )
        brand_tracker.observe(observations)
        dangers_by_frame[frame_no] = brand_tracker.danger_targets(frame_no, coordinate_segment)
        danger_segments_by_frame[frame_no] = coordinate_segment
        frame_no += 1

    # Behavior events: MOT_FARA resolves the danger point per frame from the
    # situation analyzer's fire/smoke detection — scene-transformed per its
    # own frame_to_scene when B29 scene data exists, else in pixel space. This
    # is NOT a mean: a relocating fire is tracked at its actual position each
    # frame instead of collapsed to a fabricated midpoint that was measured on
    # no frame (the old behavior breached B29's own anti-fabrication rule).
    # When no frame ever had a danger point, the resolver is None and MOT_FARA
    # cannot fire (STILLA, danger-independent, still can). The review layer's
    # reviewer-driven hazard marker (review/hazard.py) is a separate constant
    # resolver merged in at read time exactly like Phase 3's verdicts overlay;
    # it never rewrites this engine output.
    danger_for_frame = None
    if any(dangers_by_frame.values()):

        def danger_for_frame(frame_no: int, segment: int | None) -> DangerSet:
            expected_segment = danger_segments_by_frame.get(frame_no)
            if expected_segment is not None and segment != expected_segment:
                return None
            return dangers_by_frame.get(frame_no)

    behavior_events = derive_behavior_events(
        tracklet_rows,
        person_by_tracklet=person_by_tracklet,
        fps=fps,
        frame_w=frame_w,
        frame_h=frame_h,
        config=config,
        danger_for_frame=danger_for_frame,
    )

    # IRRATIONELL (Phase 4, report §4): the same tracklet trajectories, run
    # through the sub-signal ensemble in analysis/irrational.py. Precedence
    # rule — STILLA wins over IRRATIONELL — is implemented by collecting
    # every STILLA event's covered frames per tracklet from the behavior
    # events we already computed above (no second BehaviorAnalyzer replay
    # needed) and forcing those frames to non-fired in the ensemble.
    still_frames_by_tracklet: dict[int, set[int]] = defaultdict(set)
    for ev in behavior_events:
        if ev.category == CATEGORY_STILLA:
            tid = ev.evidence.get("tracklet_id")
            if tid is not None:
                frame_range = range(ev.evidence["frame_start"], ev.evidence["frame_end"] + 1)
                still_frames_by_tracklet[tid].update(frame_range)

    irrational_events = derive_irrational_events(
        tracklet_rows,
        person_by_tracklet=person_by_tracklet,
        fps=fps,
        config=_irrational_config_from_offline(config),
        still_frames_by_tracklet=still_frames_by_tracklet,
    )

    # BRAND incidents persist to video end; detector gaps update
    # last_observed_frame but do not pretend that the physical fire stopped.
    hazard_events = _brand_incidents_to_events(
        brand_tracker.incidents(end_frame=max(0, frame_no - 1), fps=fps), fps
    )

    # Merge into a single time-ordered log. Stable sort preserves the
    # within-category ordering above.
    all_events = behavior_events + irrational_events + hazard_events
    all_events.sort(key=lambda e: (e.t_start, e.category, e.event_id))
    return all_events
