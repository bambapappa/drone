"""P4 IRRATIONELL behavior: a transparent heuristic ensemble over stabilized,
body-height-normalized trajectories (architecture report §4).

Five sub-signals, each a pure function of a tracklet's own trajectory (plus,
for two of them, the concurrent kinematics of other tracklets — "the group"):

  erratic       — path tortuosity > 3 over a 10s window with mean speed
                  > 0.4 bh/s, OR heading variance above threshold
  sprint        — sustained speed > 2.5 bh/s for >= 2s AND > 2x the
                  concurrent group median speed (group-relative so a general
                  evacuation does not flag everyone — see _eval_sprint)
  counterflow   — heading > 120 deg off the dominant local group flow,
                  sustained >= 3s, when >= 3 others within a radius move
                  coherently (flow coherence > 0.7)
  oscillation   — >= 3 direction reversals between the same two areas
                  within 30s
  freeze_bolt   — STILLA-like stillness < 0.1 bh/s for >= 3s followed
                  within 2s by > 2.0 bh/s

Trajectory convention: reused verbatim from analysis/behavior.py and
analysis/events.py — foot-center position ((x0+x1)/2, y1) from P2's
tracker-adjusted xyxy, speed normalized by the person's own box height
("body heights per second"). No new coordinate convention is introduced;
positions are raw frame pixels (no persisted GMC stabilization, per
analysis/identity.py's note), exactly like STILLA/MOT_FARA's substrate.

Combination: a weighted score over the sub-signals that fired this frame,
gated by a sustained-duration requirement (report §4) — mirrors
BehaviorAnalyzer's still_since/toward_since hysteresis pattern. Every
IRRATIONELL event's evidence stores which sub-signals fired and their exact
measured values (never a bare label) — the same evidence-not-assertion
discipline as P3's assoc_audit.

Precedence (report §4, exact): STILLA (injured) wins over IRRATIONELL — a
frame the behavior analyzer already confidently flagged as sustained
stillness cannot also register as irrational, even if the ensemble's own
rolling state would otherwise qualify it (e.g. right at a freeze-and-bolt
boundary, where a genuine STILLA episode is the correct read, not a
freeze-and-bolt in progress). MOT_FARA has no precedence interaction with
either category — it is derived from an entirely independent status stream
(danger-point proximity) and this module never references it.

Determinism: pure functions over already-persisted trajectories, walked in
fixed (tracklet_id, frame_no) order, no RNG — two runs over the same P1+P2
output produce byte-identical events, mirroring every other pass's
guarantee.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    # Deferred at runtime (see derive_irrational_events) to avoid a circular
    # import: analysis.events imports this module to call
    # derive_irrational_events, so importing Event back at module load time
    # would be circular. By call time analysis.events is already initialized.
    from analysis.events import Event

CATEGORY_IRRATIONELL = "IRRATIONELL"


@dataclass
class IrrationalConfig:
    """Sub-signal thresholds (report §4). Units: body-heights/second for
    speed, degrees for angles, seconds for durations. All config-exposed via
    OfflineConfig's irr_* fields (see analysis/orchestrator.py) — this
    dataclass is the analyzer-facing translation, mirroring BehaviorConfig's
    relationship to OfflineConfig's beh_* fields.

    Two thresholds are not specified by the report and are this module's own
    documented defaults:
      - erratic_heading_circ_var: the report says "heading variance above
        threshold" without a number or a unit convention. Circular variance
        (1 - |mean unit vector|, bounded [0,1]) is used instead of raw
        degrees-variance to avoid the 359-vs-1-degree wraparound artifact a
        naive variance would have; 0.6 means "headings are scattered enough
        that there is no dominant direction," a reasonable disorientation
        signal.
      - counterflow_radius_bh: the report specifies the neighbor-count and
        coherence gates exactly but not the search radius ("others within a
        radius"). 10 body-heights is a walking-formation-scale neighborhood
        — far enough to catch a nearby group, not so far it pulls in
        unrelated people elsewhere in frame.
    """

    # Erratic path
    erratic_window_s: float = 10.0
    erratic_tortuosity: float = 3.0
    erratic_min_speed_bh: float = 0.4
    erratic_heading_circ_var: float = 0.6

    # Panic sprint
    sprint_speed_bh: float = 2.5
    sprint_time_s: float = 2.0
    sprint_group_multiple: float = 2.0

    # Counter-flow
    counterflow_angle_deg: float = 120.0
    counterflow_time_s: float = 3.0
    counterflow_min_neighbors: int = 3
    counterflow_radius_bh: float = 10.0
    counterflow_coherence: float = 0.7

    # Oscillation
    oscillation_window_s: float = 30.0
    oscillation_min_reversals: int = 3
    oscillation_min_excursion_bh: float = 1.5

    # Freeze-and-bolt
    freeze_speed_bh: float = 0.1
    freeze_time_s: float = 3.0
    bolt_speed_bh: float = 2.0
    bolt_within_s: float = 2.0
    # How long the fired flag stays "on" after a bolt is confirmed — without
    # this, a freeze-and-bolt is a single-frame blip that can never satisfy
    # the ensemble's own sustained-duration gate below. Must be >= sustain_s
    # (with margin) or a freeze-and-bolt could never promote to an event on
    # its own — every other sub-signal keeps firing every frame the
    # underlying condition holds, but a bolt is a one-instant confirmation,
    # so this hold window is what gives it the same chance to sustain.
    freeze_bolt_hold_s: float = 3.5

    # Ensemble combination
    weight_erratic: float = 1.0
    weight_sprint: float = 1.0
    weight_counterflow: float = 1.0
    weight_oscillation: float = 1.0
    weight_freeze_bolt: float = 1.0
    # Equal weights + threshold 1.0 means any single sub-signal firing is
    # sufficient — an OR-combination in effect, consistent with the report's
    # explicit recall-over-precision rationale (every flag lands in a human
    # review queue). Corroborating sub-signals raise the score further,
    # which feeds directly into confidence below.
    score_threshold: float = 1.0
    sustain_s: float = 3.0


# ---- trajectory kinematics ----


@dataclass
class _Kinematic:
    frame_no: int
    t: float
    x: float
    y: float
    box_h: float
    speed_bh: float
    heading_deg: float | None  # None: no reliable direction this frame


def _build_kinematics(rows: list[dict[str, Any]], fps: float) -> list[_Kinematic]:
    """One tracklet's rows (already sorted by frame_no) -> per-frame
    instantaneous kinematics via backward difference. Foot-center position
    and body-height normalization exactly match analysis/events.py's
    derive_behavior_events (same tracker-adjusted xyxy, same convention)."""
    out: list[_Kinematic] = []
    prev: _Kinematic | None = None
    for row in rows:
        frame_no = int(row["frame_no"])
        x0, y0, x1, y1 = (float(v) for v in row["xyxy"])
        x, y = (x0 + x1) / 2.0, y1
        box_h = max(y1 - y0, 1.0)
        t = frame_no / fps
        if prev is None:
            k = _Kinematic(frame_no, t, x, y, box_h, 0.0, None)
        else:
            dt = t - prev.t
            dx, dy = x - prev.x, y - prev.y
            dist = math.hypot(dx, dy)
            body = max((box_h + prev.box_h) / 2.0, 1.0)
            speed_bh = dist / dt / body if dt > 0 else 0.0
            heading = math.degrees(math.atan2(dy, dx)) % 360.0 if dist > 1e-6 and dt > 0 else None
            k = _Kinematic(frame_no, t, x, y, box_h, speed_bh, heading)
        out.append(k)
        prev = k
    return out


# ---- circular statistics (angular data — avoids 359-vs-1-degree wraparound) ----


def _circular_variance(headings_deg: list[float]) -> float:
    """1 - |mean unit vector|. 0 = perfectly aligned, 1 = uniformly scattered."""
    if not headings_deg:
        return 0.0
    sin_sum = sum(math.sin(math.radians(h)) for h in headings_deg)
    cos_sum = sum(math.cos(math.radians(h)) for h in headings_deg)
    r = math.hypot(sin_sum, cos_sum) / len(headings_deg)
    return 1.0 - r


def _circular_mean_deg(headings_deg: list[float]) -> float | None:
    if not headings_deg:
        return None
    sin_sum = sum(math.sin(math.radians(h)) for h in headings_deg)
    cos_sum = sum(math.cos(math.radians(h)) for h in headings_deg)
    if abs(sin_sum) < 1e-9 and abs(cos_sum) < 1e-9:
        return None
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angular difference, in [0, 180]."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2] if s else 0.0


# ---- per-tracklet ensemble analyzer ----


class _IrrationalTrackletAnalyzer:
    """Walks one tracklet's kinematics in frame order, evaluating all five
    sub-signals plus the weighted-score ensemble. Mirrors BehaviorAnalyzer's
    incremental deque + since-timestamp idiom (analysis/behavior.py) —
    everything here is O(1) amortized per frame, no windowed re-scan."""

    def __init__(self, cfg: IrrationalConfig):
        self.cfg = cfg
        self._erratic_hist: deque = deque()
        self._sprint_since: float | None = None
        self._counterflow_since: float | None = None
        self._osc_pivot: tuple[float, float] | None = None
        self._osc_far: tuple[float, float] | None = None
        self._osc_far_dist_bh: float = 0.0
        self._osc_reversals: deque = deque()
        self._still_since: float | None = None
        self._last_still_t: float | None = None
        self._pending_freeze_end: float | None = None
        self._pending_freeze_duration: float = 0.0
        self._freeze_bolt_hold_until: float | None = None
        self._freeze_bolt_last_detail: dict[str, Any] | None = None
        self._score_since: float | None = None

    def update(self, k: _Kinematic, neighbors: list[_Kinematic]) -> tuple[bool, dict[str, Any]]:
        cfg = self.cfg
        t = k.t
        fired: dict[str, dict[str, Any]] = {}

        d = self._eval_erratic(k)
        if d is not None:
            fired["erratic"] = d
        d = self._eval_sprint(k, neighbors, t)
        if d is not None:
            fired["sprint"] = d
        d = self._eval_counterflow(k, neighbors, t)
        if d is not None:
            fired["counterflow"] = d
        d = self._eval_oscillation(k, t)
        if d is not None:
            fired["oscillation"] = d
        d = self._eval_freeze_bolt(k, t)
        if d is not None:
            fired["freeze_bolt"] = d

        weights = {
            "erratic": cfg.weight_erratic,
            "sprint": cfg.weight_sprint,
            "counterflow": cfg.weight_counterflow,
            "oscillation": cfg.weight_oscillation,
            "freeze_bolt": cfg.weight_freeze_bolt,
        }
        score = sum(weights[name] for name in fired)

        if score >= cfg.score_threshold:
            if self._score_since is None:
                self._score_since = t
        else:
            self._score_since = None

        combined_fired = self._score_since is not None and (t - self._score_since) >= cfg.sustain_s
        if not combined_fired:
            return False, {}
        return True, {"score": round(score, 3), "sub_signals": fired}

    # ---- sub-signal 1: erratic path ----

    def _eval_erratic(self, k: _Kinematic) -> dict[str, Any] | None:
        cfg = self.cfg
        self._erratic_hist.append((k.t, k.x, k.y, k.box_h, k.heading_deg, k.speed_bh))
        while self._erratic_hist and k.t - self._erratic_hist[0][0] > cfg.erratic_window_s:
            self._erratic_hist.popleft()
        hist = self._erratic_hist
        if len(hist) < 4:
            return None
        span = hist[-1][0] - hist[0][0]
        if span < cfg.erratic_window_s * 0.5:
            # Require at least half the window filled — otherwise a
            # tracklet's first few samples produce noisy tortuosity.
            return None

        path_bh = 0.0
        for i in range(1, len(hist)):
            _, x0, y0, h0, _, _ = hist[i - 1]
            _, x1, y1, h1, _, _ = hist[i]
            body = max((h0 + h1) / 2.0, 1.0)
            path_bh += math.hypot(x1 - x0, y1 - y0) / body
        ref_body = max(_median([h for _, _, _, h, _, _ in hist]), 1.0)
        net_bh = math.hypot(hist[-1][1] - hist[0][1], hist[-1][2] - hist[0][2]) / ref_body
        tortuosity = path_bh / max(net_bh, 1e-6)
        mean_speed = _mean([sp for _, _, _, _, _, sp in hist])
        headings = [h for _, _, _, _, h, _ in hist if h is not None]
        circ_var = _circular_variance(headings) if len(headings) >= 3 else 0.0

        tortuosity_fires = tortuosity > cfg.erratic_tortuosity and mean_speed > cfg.erratic_min_speed_bh
        heading_fires = circ_var > cfg.erratic_heading_circ_var
        if not (tortuosity_fires or heading_fires):
            return None
        return {
            "tortuosity": round(tortuosity, 2),
            "mean_speed_bh": round(mean_speed, 3),
            "heading_circ_var": round(circ_var, 3),
            "window_s": round(span, 1),
        }

    # ---- sub-signal 2: panic sprint ----

    def _eval_sprint(self, k: _Kinematic, neighbors: list[_Kinematic], t: float) -> dict[str, Any] | None:
        cfg = self.cfg
        other_speeds = [n.speed_bh for n in neighbors]
        # Group-relative on purpose (report §4): comparing only against an
        # absolute speed threshold would flag every runner in a genuine
        # mass evacuation. Requiring > 2x the *concurrent* group median
        # isolates the person moving abnormally relative to what everyone
        # else is doing right now, not just "moving fast." With nobody else
        # in frame there is no group to be relative to, so the signal
        # cannot fire (a solo sprinter needs the erratic/oscillation
        # solo signals instead — report §4's own tradeoff note).
        group_median = _median(other_speeds) if other_speeds else None
        qualifies = (
            group_median is not None
            and k.speed_bh > cfg.sprint_speed_bh
            and k.speed_bh > cfg.sprint_group_multiple * group_median
        )
        if qualifies:
            if self._sprint_since is None:
                self._sprint_since = t
        else:
            self._sprint_since = None
        if self._sprint_since is None or (t - self._sprint_since) < cfg.sprint_time_s:
            return None
        return {
            "speed_bh": round(k.speed_bh, 3),
            "group_median_speed_bh": round(group_median, 3),
            "duration_s": round(t - self._sprint_since, 1),
        }

    # ---- sub-signal 3: counter-flow ----

    def _eval_counterflow(
        self, k: _Kinematic, neighbors: list[_Kinematic], t: float
    ) -> dict[str, Any] | None:
        cfg = self.cfg
        radius_px = cfg.counterflow_radius_bh * max(k.box_h, 1.0)
        near = [
            n
            for n in neighbors
            if n.heading_deg is not None and math.hypot(n.x - k.x, n.y - k.y) <= radius_px
        ]
        qualifies = False
        detail: dict[str, Any] | None = None
        if len(near) >= cfg.counterflow_min_neighbors and k.heading_deg is not None:
            headings = [n.heading_deg for n in near]
            coherence = 1.0 - _circular_variance(headings)
            if coherence > cfg.counterflow_coherence:
                dominant = _circular_mean_deg(headings)
                if dominant is not None:
                    angle_diff = _angle_diff_deg(k.heading_deg, dominant)
                    if angle_diff > cfg.counterflow_angle_deg:
                        qualifies = True
                        detail = {
                            "angle_deg": round(angle_diff, 1),
                            "dominant_flow_deg": round(dominant, 1),
                            "coherence": round(coherence, 3),
                            "neighbors": len(near),
                        }
        if qualifies:
            if self._counterflow_since is None:
                self._counterflow_since = t
        else:
            self._counterflow_since = None
        if self._counterflow_since is None or (t - self._counterflow_since) < cfg.counterflow_time_s:
            return None
        assert detail is not None
        detail["duration_s"] = round(t - self._counterflow_since, 1)
        return detail

    # ---- sub-signal 4: oscillation ----

    def _eval_oscillation(self, k: _Kinematic, t: float) -> dict[str, Any] | None:
        cfg = self.cfg
        body = max(k.box_h, 1.0)
        if self._osc_pivot is None:
            self._osc_pivot = (k.x, k.y)
            self._osc_far = (k.x, k.y)
            self._osc_far_dist_bh = 0.0
            return None
        assert self._osc_far is not None
        d_pivot = math.hypot(k.x - self._osc_pivot[0], k.y - self._osc_pivot[1]) / body
        if d_pivot > self._osc_far_dist_bh:
            # Still moving away from the pivot — extend the excursion.
            self._osc_far = (k.x, k.y)
            self._osc_far_dist_bh = d_pivot
        else:
            # Retreating from the farthest point reached — a reversal once
            # the retreat itself exceeds the minimum excursion (this is the
            # standard swing-high/low zigzag detector, applied to 2D via
            # Euclidean distance from a pivot rather than a signed 1D
            # value — each alternation visits near `pivot` then near `far`,
            # i.e. "the same two areas" per the report's wording).
            d_far = math.hypot(k.x - self._osc_far[0], k.y - self._osc_far[1]) / body
            if d_far >= cfg.oscillation_min_excursion_bh:
                self._osc_reversals.append(t)
                self._osc_pivot = self._osc_far
                self._osc_far = (k.x, k.y)
                self._osc_far_dist_bh = math.hypot(k.x - self._osc_pivot[0], k.y - self._osc_pivot[1]) / body
        while self._osc_reversals and t - self._osc_reversals[0] > cfg.oscillation_window_s:
            self._osc_reversals.popleft()
        if len(self._osc_reversals) < cfg.oscillation_min_reversals:
            return None
        return {
            "reversals": len(self._osc_reversals),
            "window_s": round(t - self._osc_reversals[0], 1),
        }

    # ---- sub-signal 5: freeze-and-bolt ----

    def _eval_freeze_bolt(self, k: _Kinematic, t: float) -> dict[str, Any] | None:
        cfg = self.cfg
        if k.speed_bh < cfg.freeze_speed_bh:
            if self._still_since is None:
                self._still_since = t
            self._last_still_t = t
        else:
            if (
                self._still_since is not None
                and self._last_still_t is not None
                and (self._last_still_t - self._still_since) >= cfg.freeze_time_s
            ):
                self._pending_freeze_end = self._last_still_t
                self._pending_freeze_duration = self._last_still_t - self._still_since
            self._still_since = None

        if self._pending_freeze_end is not None:
            gap = t - self._pending_freeze_end
            if gap > cfg.bolt_within_s:
                self._pending_freeze_end = None  # window expired, no bolt
            elif k.speed_bh > cfg.bolt_speed_bh:
                self._freeze_bolt_hold_until = t + cfg.freeze_bolt_hold_s
                self._freeze_bolt_last_detail = {
                    "freeze_duration_s": round(self._pending_freeze_duration, 1),
                    "reaction_gap_s": round(gap, 1),
                    "bolt_speed_bh": round(k.speed_bh, 3),
                }
                self._pending_freeze_end = None  # consumed

        if self._freeze_bolt_hold_until is not None and t <= self._freeze_bolt_hold_until:
            return self._freeze_bolt_last_detail
        return None


# ---- trajectory quality (confidence input) ----


def _trajectory_quality(rows: list[dict[str, Any]], person_id: int | None, fps: float) -> float:
    """0..1 quality score from track length, detection density, and ID
    stability — the report's "confidence = margin * trajectory quality"
    factor. A short, sparse, or P3-unconfirmed tracklet is weaker evidence
    for an irrational-behavior claim than a long, dense, identity-confirmed
    one, even at the same sub-signal margin.

      length_score:    full credit at >=10s of track history.
      density_score:    fraction of the tracklet's frame span that actually
                        has a row (gaps mean the tracker interpolated/lost
                        the target rather than seeing it continuously).
      stability_score:  1.0 if P3 confirmed this tracklet into a person
                        (fewer identity fragments feeding the signal), 0.7
                        for a tracklet-only track (P3 didn't run) — a
                        documented simplification, not a precise stability
                        measure (that would need cross-tracklet audit data
                        this function isn't given).
    """
    frame_nos = sorted(int(r["frame_no"]) for r in rows)
    if not frame_nos:
        return 0.0
    span_frames = frame_nos[-1] - frame_nos[0] + 1
    duration_s = span_frames / fps if fps > 0 else 0.0
    length_score = min(1.0, duration_s / 10.0)
    density_score = len(frame_nos) / span_frames if span_frames > 0 else 0.0
    stability_score = 1.0 if person_id is not None else 0.7
    return max(0.0, min(1.0, (length_score + density_score + stability_score) / 3.0))


def _confidence(max_score: float, threshold: float, quality: float) -> float:
    """Floor + scaled-margin style, matching STILLA/MOT_FARA's confidence
    formula in analysis/events.py (never claims certainty; a barely-
    qualifying single-signal event floors at 0.3, a strong multi-signal
    high-quality-track event approaches 1.0)."""
    margin_norm = max(0.0, min(1.0, (max_score - threshold) / max(threshold, 1e-6)))
    return max(0.0, min(1.0, 0.3 + 0.7 * margin_norm * quality))


# ---- evidence formatting ----

_HEADLINE_KEY = {
    "erratic": "tortuosity",
    "sprint": "speed_bh",
    "counterflow": "angle_deg",
    "oscillation": "reversals",
    "freeze_bolt": "bolt_speed_bh",
}


def _stronger_detail(name: str, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Keep whichever detail snapshot has the larger headline value, so a
    span's evidence reflects the strongest observed instance of a
    sub-signal rather than just the first frame it fired on."""
    key = _HEADLINE_KEY.get(name)
    if key is None:
        return a
    return b if b.get(key, 0) > a.get(key, 0) else a


def _format_summary(by_name: dict[str, dict[str, Any]]) -> str:
    """One-line evidence-not-assertion summary, e.g. "counter-flow (155.0
    deg, 4.2s) + erratic (tortuosity 4.1)" — report §4's exact phrasing.
    Built only from the measured values already in evidence.sub_signals."""
    parts = []
    if "counterflow" in by_name:
        d = by_name["counterflow"]
        parts.append(f"counter-flow ({d['angle_deg']} deg, {d.get('duration_s', 0)}s)")
    if "erratic" in by_name:
        d = by_name["erratic"]
        parts.append(f"erratic (tortuosity {d['tortuosity']})")
    if "sprint" in by_name:
        d = by_name["sprint"]
        parts.append(f"panic-sprint ({d['speed_bh']} bh/s, {d.get('duration_s', 0)}s)")
    if "oscillation" in by_name:
        d = by_name["oscillation"]
        parts.append(f"oscillation ({d['reversals']} reversals)")
    if "freeze_bolt" in by_name:
        d = by_name["freeze_bolt"]
        parts.append(f"freeze-and-bolt ({d['freeze_duration_s']}s still -> {d['bolt_speed_bh']} bh/s)")
    return " + ".join(parts) if parts else "irrational behavior (unspecified)"


def _irr_event_id(seq: int) -> str:
    return f"{CATEGORY_IRRATIONELL.lower()}-{seq:06d}"


# ---- top-level derivation ----


def derive_irrational_events(
    tracklet_rows: Iterable[dict[str, Any]],
    person_by_tracklet: dict[int, int],
    fps: float,
    config: IrrationalConfig,
    still_frames_by_tracklet: dict[int, set[int]] | None = None,
) -> "list[Event]":
    """Replay the five sub-signals + ensemble over a tracklet table and diff
    into IRRATIONELL events, one per contiguous confidently-flagged span
    (mirrors analysis/events.py's STILLA/MOT_FARA span diffing exactly).

    `still_frames_by_tracklet` implements the STILLA-wins-over-IRRATIONELL
    precedence rule (report §4): frame numbers already covered by a STILLA
    event for that tracklet are forced to non-fired here, regardless of what
    the ensemble's own rolling state would otherwise say. Pass None (the
    default) to derive IRRATIONELL with no such suppression, e.g. in tests
    that only exercise the sub-signals themselves.
    """
    # Deferred import — see the TYPE_CHECKING note at the top of this file.

    if fps <= 0:
        return []
    still_frames_by_tracklet = still_frames_by_tracklet or {}

    by_tracklet: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in tracklet_rows:
        by_tracklet[int(row["tracklet_id"])].append(row)
    for tid in by_tracklet:
        by_tracklet[tid].sort(key=lambda r: int(r["frame_no"]))

    kinematics_by_tid: dict[int, list[_Kinematic]] = {
        tid: _build_kinematics(rows, fps) for tid, rows in by_tracklet.items()
    }
    # frame_no -> [(tracklet_id, kinematic), ...] for the group sub-signals
    # (panic sprint's group median, counter-flow's neighbor flow).
    frame_index: dict[int, list[tuple[int, _Kinematic]]] = defaultdict(list)
    for tid, kins in kinematics_by_tid.items():
        for k in kins:
            frame_index[k.frame_no].append((tid, k))

    events: list[Event] = []
    seq = 0
    for tid in sorted(kinematics_by_tid.keys()):
        kins = kinematics_by_tid[tid]
        still_frames = still_frames_by_tracklet.get(tid, set())
        person_id = person_by_tracklet.get(tid)
        quality = _trajectory_quality(by_tracklet[tid], person_id, fps)
        analyzer = _IrrationalTrackletAnalyzer(config)
        timeline: list[tuple[int, bool, dict[str, Any]]] = []
        for k in kins:
            neighbors = [nk for ntid, nk in frame_index[k.frame_no] if ntid != tid]
            fired, evidence = analyzer.update(k, neighbors)
            if k.frame_no in still_frames:
                fired, evidence = False, {}
            timeline.append((k.frame_no, fired, evidence))
        new_events, seq = _diff_irrational_timeline(
            timeline,
            tracklet_id=tid,
            person_id=person_id,
            fps=fps,
            quality=quality,
            config=config,
            seq_start=seq,
        )
        events.extend(new_events)
    events.sort(key=lambda e: (e.t_start, e.event_id))
    return events


def _diff_irrational_timeline(
    timeline: list[tuple[int, bool, dict[str, Any]]],
    tracklet_id: int,
    person_id: int | None,
    fps: float,
    quality: float,
    config: IrrationalConfig,
    seq_start: int,
) -> tuple["list[Event]", int]:
    from analysis.events import Event

    events: list[Event] = []
    seq = seq_start
    span_start_frame: int | None = None
    span_evidence: list[dict[str, Any]] = []
    span_max_score = 0.0

    def flush(end_frame: int) -> None:
        nonlocal seq
        if span_start_frame is None:
            return
        duration_s = (end_frame - span_start_frame + 1) / fps
        by_name: dict[str, dict[str, Any]] = {}
        for snapshot in span_evidence:
            for name, detail in snapshot.get("sub_signals", {}).items():
                if name not in by_name:
                    by_name[name] = detail
                else:
                    by_name[name] = _stronger_detail(name, by_name[name], detail)
        evidence = {
            "tracklet_id": tracklet_id,
            "frame_start": span_start_frame,
            "frame_end": end_frame,
            "duration_s": round(duration_s, 3),
            "sub_signals": by_name,
            "summary": _format_summary(by_name),
        }
        confidence = _confidence(span_max_score, config.score_threshold, quality)
        events.append(
            Event(
                event_id=_irr_event_id(seq),
                category=CATEGORY_IRRATIONELL,
                person_id=person_id,
                t_start=span_start_frame / fps,
                t_end=(end_frame + 1) / fps,
                confidence=confidence,
                evidence=evidence,
            )
        )
        seq += 1

    for frame_no, fired, evidence in timeline:
        if fired:
            if span_start_frame is None:
                span_start_frame = frame_no
                span_evidence = []
                span_max_score = 0.0
            span_evidence.append(evidence)
            span_max_score = max(span_max_score, evidence.get("score", 0.0))
        else:
            if span_start_frame is not None:
                flush(frame_no - 1)
            span_start_frame = None
    if span_start_frame is not None and timeline:
        flush(timeline[-1][0])
    return events, seq
