"""P3 identity: global tracklet association into persons.

The tracker (P2) keeps an ID through short occlusions, but a track that dies
(person leaves frame, long occlusion) comes back as a *new* tracklet. The
live registry.py re-identifies online-greedily — its decisions are
irrevocable and order-dependent, and a later better match can never merge
two IDs already split (the scout's Q3 finding). Offline we have the whole
video, so we do **global tracklet association**: aggregate each tracklet's
per-detection appearance into a centroid, then run constrained agglomerative
clustering to merge tracklet pairs into persons.

Three gates, in order of strength:

1. **Hard temporal-overlap exclusion.** Two tracklets that share even one
   frame are *never* the same person — a single physical body cannot produce
   two simultaneous tracker boxes. This is an impossible constraint in a live
   system (you never know a tracklet's full frame set until the video ends)
   but trivial offline, where the complete frame index is on disk. It is the
   single biggest correctness lever the offline tool has over the live one
   for identity, and it is why global association is structurally superior to
   online-greedy here. (registry.py cannot enforce it: it only ever sees the
   frame it's in.)

2. **Spatio-temporal plausibility.** Generalizes the live registry's gate
   (`registry.py:_match_lost`): a re-entering tracklet must have appeared
   within `max_dist_frac × diag × (1 + gap_s)` of where the earlier tracklet
   was last seen. The `(1 + gap)` term lets a person roam further the longer
   they were off-screen, so the gate is binding for short gaps (a re-entry
   0.5s later must be spatially close) and relaxes to "anywhere in frame" for
   long gaps (where appearance must carry the decision alone). NB: positions
   are in raw frame pixels (no GMC stabilization persisted in Phase 0), so
   this gate is a plausibility filter, not a motion model — it rejects absurd
   teleports; appearance + temporal exclusion do the real identity work.

3. **Appearance similarity.** Cosine of the per-method embedding centroids,
   best same-method wins (an osnet vector and an hsv vector are different-
   dimensional and not comparable — P3 never compares across methods).

**Determinism.** Clustering is fully deterministic: no RNG is consumed, and
candidate merges are evaluated in a fixed order (similarity descending, then
smallest bridging gap, then lowest tracklet-id pair) so two runs over the
same P1+P2 output produce byte-identical persons + assoc_audit. This is the
Phase 1 analogue of Phase 0's P1/P2 determinism guarantee.

**Honest counting.** Same-clothing confusion is a fundamental limit of any
appearance method (DECISIONS B4), not a bug to paper over. So the count is
never a single confident number: confirmed persons plus an uncertainty band
from below-threshold / gate-blocked near-merges ("N unika, varav M osäkra
sammanslagningar"). Every merge and every notable blocked near-merge is
recorded in `assoc_audit` — that trail is load-bearing, not cosmetic: a
later-phase reviewer reads it to see *why* two tracklets became one person,
and a manual split/merge correction acts on exactly these entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analysis.orchestrator import OfflineConfig


@dataclass
class TrackletProfile:
    """A tracklet's aggregated identity features, built once from P2's
    tracklet rows joined to P1's per-detection embeddings via det_id."""

    tracklet_id: int
    cls: str
    frame_start: int
    frame_end: int
    first_seen: float  # video seconds
    last_seen: float
    frames: set[int] = field(default_factory=set)
    # Per-method L2-normalized centroid (count-weighted mean of per-detection
    # vectors, renormalized). A tracklet may have both if some of its crops
    # were above the ReID floor and some below — P3 compares within a method.
    centroids: dict[str, np.ndarray] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    # Center of the first / last detection box (raw frame pixels) — the two
    # endpoints the spatio-temporal gate measures re-entry distance across.
    start_center: tuple[float, float] = (0.0, 0.0)
    end_center: tuple[float, float] = (0.0, 0.0)


@dataclass
class AuditEvent:
    """One association decision, recorded for review. `rule` is either
    "merged" (this pair became one person) or "blocked:<reason>" (a notable
    near-merge that the gates rejected, feeding the uncertainty band)."""

    tracklet_a: int
    tracklet_b: int
    appearance_sim: float
    method: str
    gap_s: float
    dist: float
    dist_limit: float
    rule: str

    def to_dict(self) -> dict:
        return {
            "tracklet_a": self.tracklet_a,
            "tracklet_b": self.tracklet_b,
            "appearance_sim": round(self.appearance_sim, 6),
            "method": self.method,
            "gap_s": round(self.gap_s, 4),
            "dist": round(self.dist, 2),
            "dist_limit": round(self.dist_limit, 2),
            "rule": self.rule,
        }


@dataclass
class PersonRecord:
    person_id: int
    tracklet_ids: list[int]
    embedding_centroids: dict[str, list[float]]
    embedding_counts: dict[str, int]
    first_seen: float
    last_seen: float
    confirmation_state: str
    assoc_audit: list[dict]


@dataclass
class AssociationResult:
    persons: list[PersonRecord]
    confirmed_count: int
    uncertain_merges: int  # near-merges in the honesty band


# ---------------------------------------------------------------------------
# Feature aggregation
# ---------------------------------------------------------------------------


def build_tracklet_profiles(
    tracklet_rows: list[dict],
    detection_by_id: dict[int, dict],
    fps: float,
) -> list[TrackletProfile]:
    """Aggregate P2's per-(tracklet, frame) rows + P1's per-detection
    embeddings (joined via det_id) into one TrackletProfile per tracklet.

    Deterministic ordering: profiles are returned sorted by tracklet_id, so
    downstream clustering sees a stable input order regardless of JSONL
    append timing.
    """
    by_id: dict[int, TrackletProfile] = {}
    # Accumulators for count-weighted centroids (sum then normalize once).
    sums: dict[int, dict[str, np.ndarray]] = {}
    # Track min/max frame per tracklet for start/end centers (robust to any
    # ordering quirk; P2 writes rows in ascending frame order, but this makes
    # the centers a pure function of the data, not the write order).
    first_fno: dict[int, int] = {}
    last_fno: dict[int, int] = {}
    centers: dict[int, dict[int, tuple[float, float]]] = {}

    for row in tracklet_rows:
        tid = row["tracklet_id"]
        fno = row["frame_no"]
        det = detection_by_id.get(row["det_id"])
        prof = by_id.get(tid)
        if prof is None:
            prof = TrackletProfile(
                tracklet_id=tid,
                cls=row.get("cls", "person"),
                frame_start=fno,
                frame_end=fno,
                first_seen=fno / max(fps, 0.001),
                last_seen=fno / max(fps, 0.001),
            )
            by_id[tid] = prof
            sums[tid] = {}
            centers[tid] = {}
        prof.frames.add(fno)
        prof.frame_start = min(prof.frame_start, fno)
        prof.frame_end = max(prof.frame_end, fno)

        x0, y0, x1, y1 = row["xyxy"]
        centers[tid][fno] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        if tid not in first_fno or fno < first_fno[tid]:
            first_fno[tid] = fno
        if tid not in last_fno or fno > last_fno[tid]:
            last_fno[tid] = fno

        if det is not None:
            emb = det.get("embedding")
            method = det.get("embedding_method")
            if emb and method:
                v = np.asarray(emb, dtype=np.float64)
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
                    acc = sums[tid].get(method)
                    if acc is None:
                        sums[tid][method] = v.copy()
                    else:
                        sums[tid][method] += v
                    prof.counts[method] = prof.counts.get(method, 0) + 1

    fps_safe = max(fps, 0.001)
    for tid, prof in by_id.items():
        for method, acc in sums[tid].items():
            n = np.linalg.norm(acc)
            prof.centroids[method] = acc / n if n > 0 else acc
        prof.first_seen = prof.frame_start / fps_safe
        prof.last_seen = prof.frame_end / fps_safe
        prof.start_center = centers[tid][first_fno[tid]]
        prof.end_center = centers[tid][last_fno[tid]]
    return [by_id[k] for k in sorted(by_id)]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class _Cluster:
    """Mutable cluster during agglomerative merging."""

    def __init__(self, profile: TrackletProfile):
        self.profiles: list[TrackletProfile] = [profile]
        self.tracklet_ids: set[int] = {profile.tracklet_id}
        self.frames: set[int] = set(profile.frames)
        self.centroids: dict[str, np.ndarray] = {m: v.copy() for m, v in profile.centroids.items()}
        self.counts: dict[str, int] = dict(profile.counts)

    def absorb(self, other: "_Cluster") -> None:
        """Merge `other` into self: count-weighted centroid update."""
        for m, v in other.centroids.items():
            own = self.centroids.get(m)
            oc = self.counts.get(m, 0)
            nc = other.counts.get(m, 0)
            if own is None:
                self.centroids[m] = v.copy()
                self.counts[m] = nc
            else:
                # count-weighted blend, then renormalize
                blended = (own * oc + v * nc) / max(oc + nc, 1)
                nrm = np.linalg.norm(blended)
                self.centroids[m] = blended / nrm if nrm > 0 else blended
                self.counts[m] = oc + nc
        self.frames |= other.frames
        self.profiles.extend(other.profiles)
        self.tracklet_ids |= other.tracklet_ids


def _best_appearance_sim(a: _Cluster, b: _Cluster) -> tuple[float, str]:
    """Best same-method cosine between two clusters' centroids.

    Returns (similarity, method). -inf / "" when no shared embedding method
    exists (an osnet vector and an hsv vector are not comparable), which
    means appearance cannot support a merge — the pair stays apart.
    """
    best = -1.0
    best_method = ""
    for m, va in a.centroids.items():
        vb = b.centroids.get(m)
        if vb is None:
            continue
        sim = float(np.dot(va, vb))  # both L2-normalized → cosine
        if sim > best:
            best, best_method = sim, m
    return best, best_method


def _temporal_overlap(a: _Cluster, b: _Cluster) -> bool:
    """True if any tracklet in a and any in b share a frame. The hard
    offline-only exclusion: two simultaneously-visible tracklets are never
    the same person."""
    # set.isdisjoint is the fast path; non-disjoint => overlap.
    return not a.frames.isdisjoint(b.frames)


def _bridging_pair(a: _Cluster, b: _Cluster, fps: float) -> tuple[float, float, float] | None:
    """Find the temporally-closest non-overlapping cross-pair of tracklets.

    Returns (gap_s, dist_pixels, dist_limit_pixels) for the bridging pair, or
    None if every cross-pair overlaps (caller should have already excluded
    overlap at the cluster level, so this is a defensive None). `dist_limit`
    uses the spatio-temporal gate `max_dist_frac × diag × (1 + gap)` — but
    diag is applied by the caller; here we return the raw gap and let the
    caller compute the limit, so this stays diag-agnostic.
    """
    best: tuple[float, float] | None = None  # (gap, dist)
    for pa in a.profiles:
        for pb in b.profiles:
            if not pa.frames.isdisjoint(pb.frames):
                continue  # overlap — not a valid bridge
            if pa.last_seen <= pb.first_seen:
                gap = pb.first_seen - pa.last_seen
                dist = float(
                    np.hypot(pb.start_center[0] - pa.end_center[0], pb.start_center[1] - pa.end_center[1])
                )
            else:
                gap = pa.first_seen - pb.last_seen
                dist = float(
                    np.hypot(pa.start_center[0] - pb.end_center[0], pa.start_center[1] - pb.end_center[1])
                )
            if best is None or gap < best[0]:
                best = (gap, dist)
    if best is None:
        return None
    return best[0], best[1], 0.0  # dist_limit filled by caller (needs diag)


def associate(
    profiles: list[TrackletProfile],
    config: OfflineConfig,
    frame_diag: float,
) -> AssociationResult:
    """Run constrained agglomerative clustering over tracklet profiles.

    Merges tracklet pairs into persons gated by (1) hard temporal-overlap
    exclusion, (2) spatio-temporal plausibility, (3) appearance similarity.
    Fully deterministic: no RNG; fixed evaluation order.

    Returns persons (sorted by first_seen then min tracklet_id, ids from 1),
    the confirmed-person count, and the uncertainty-band near-merge count.
    """
    clusters: dict[int, _Cluster] = {p.tracklet_id: _Cluster(p) for p in profiles}
    # audit accumulated per surviving cluster id (keyed by the lowest member
    # tracklet_id, which is stable across merges since absorb never renames).
    audit: dict[int, list[AuditEvent]] = {cid: [] for cid in clusters}
    uncertain_seen: set[tuple[int, int]] = set()

    sim_thresh = config.p3_sim_thresh
    uncertain_lo = sim_thresh - config.p3_uncertain_margin

    while True:
        # Evaluate all cluster pairs; pick the best FEASIBLE merge.
        best_merge: tuple[float, float, int, int, float, float, float, str] | None = None
        # (sim, -gap, a_id, b_id, gap, dist, dist_limit, method) — sorted by
        # sim desc then gap asc then id pair; encoded via direct comparison.
        ids = sorted(clusters)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                aid, bid = ids[i], ids[j]
                ca, cb = clusters[aid], clusters[bid]
                sim, method = _best_appearance_sim(ca, cb)
                if method == "":
                    continue  # no comparable embedding → appearance can't support merge

                # Hard temporal-overlap exclusion FIRST: if any tracklet in a
                # shares a frame with any in b, they are never the same person.
                # Record as uncertain only if appearance was high (same-clothing
                # twins seen together — exactly the B4 ambiguity the honesty
                # band exists to surface).
                if _temporal_overlap(ca, cb):
                    if sim >= uncertain_lo:
                        uncertain_seen.add((aid, bid))
                    continue

                bridge = _bridging_pair(ca, cb, 0.0)
                if bridge is None:
                    continue
                gap, dist, _ = bridge
                dist_limit = config.p3_max_dist_frac * frame_diag * (1.0 + gap)

                # Record near-merges in the uncertainty band: appearance
                # promising but a gate rejected the merge.
                if sim >= uncertain_lo and (
                    gap < config.p3_min_gap_s
                    or gap > config.p3_max_gap_s
                    or dist > dist_limit
                    or sim < sim_thresh
                ):
                    uncertain_seen.add((aid, bid))

                if sim < sim_thresh:
                    continue
                if gap < config.p3_min_gap_s or gap > config.p3_max_gap_s:
                    continue
                if dist > dist_limit:
                    continue

                # Feasible merge candidate. Deterministic ranking.
                cand = (sim, -gap, aid, bid, gap, dist, dist_limit, method)
                if best_merge is None or _merge_better(cand, best_merge):
                    best_merge = cand

        if best_merge is None:
            break

        sim, neg_gap, aid, bid, gap, dist, dist_limit, method = best_merge
        # Perform the merge: bid absorbed into aid (keep the lower id as the
        # stable audit key). Record the merge event in aid's audit.
        clusters[aid].absorb(clusters[bid])
        audit[aid].append(
            AuditEvent(
                tracklet_a=aid,
                tracklet_b=bid,
                appearance_sim=sim,
                method=method,
                gap_s=gap,
                dist=dist,
                dist_limit=dist_limit,
                rule="merged",
            )
        )
        audit[aid].extend(audit.pop(bid, []))
        del clusters[bid]

    # Emit persons in deterministic order: first_seen, then min tracklet_id.
    surviving = list(clusters.values())

    def sort_key(c: _Cluster) -> tuple[float, int]:
        fs = min(p.first_seen for p in c.profiles)
        return (fs, min(c.tracklet_ids))

    surviving.sort(key=sort_key)

    persons: list[PersonRecord] = []
    confirmed = 0
    for pid, cl in enumerate(surviving, start=1):
        first_seen = min(p.first_seen for p in cl.profiles)
        last_seen = max(p.last_seen for p in cl.profiles)
        state = "confirmed" if (last_seen - first_seen) >= config.p3_confirm_s else "transient"
        if state == "confirmed":
            confirmed += 1
        persons.append(
            PersonRecord(
                person_id=pid,
                tracklet_ids=sorted(cl.tracklet_ids),
                embedding_centroids={m: [round(float(x), 6) for x in v] for m, v in cl.centroids.items()},
                embedding_counts=dict(cl.counts),
                first_seen=round(first_seen, 4),
                last_seen=round(last_seen, 4),
                confirmation_state=state,
                assoc_audit=[e.to_dict() for e in sorted(audit[min(cl.tracklet_ids)], key=_audit_sort)],
            )
        )
    return AssociationResult(persons=persons, confirmed_count=confirmed, uncertain_merges=len(uncertain_seen))


def _merge_better(a: tuple, b: tuple) -> bool:
    """Compare two merge candidates by (sim desc, gap asc, id pair asc)."""
    # a = (sim, neg_gap, aid, bid, ...); higher sim wins; tie → smaller gap
    # (neg_gap less negative) → tie → smaller (aid, bid).
    if a[0] != b[0]:
        return a[0] > b[0]
    if a[1] != b[1]:
        return a[1] > b[1]  # -gap larger == gap smaller
    return (a[2], a[3]) < (b[2], b[3])


def _audit_sort(e: AuditEvent) -> tuple:
    return (e.rule != "merged", -e.appearance_sim, e.tracklet_a, e.tracklet_b)
