"""Retroactive hazard-marker recompute (Phase 4, report §5.1).

The reviewer can place or move a hazard marker *during review* and see
MOT_FARA events recompute across the entire timeline in seconds — a
genuinely new capability the live tool could never offer (it could only flag
from the moment an operator tapped). This is cheap because trajectories are
already persisted and MOT_FARA derivation is a pure function over them
(analysis/events.py's derive_behavior_events): no re-running P1-P3, no
re-detection, just a replay of the already-cheap BehaviorAnalyzer heuristic
over P2's already-loaded tracklet table.

**Why this doesn't violate interface rule 2** ("review/ never imports the
engine"). That rule guards the review layer against invoking P1/P2/P3's
heavy, stateful, model-driven passes (detection inference, tracking,
identity clustering) — the actual "engine." `derive_behavior_events` is not
that: it is a pure function of already-persisted trajectories with no model,
no state carried between calls, and no I/O beyond reading a JSONL table the
review layer already reads for the overlay canvas. The architecture report's
own scope-discipline note (§6) draws exactly this line: "any feature that
can be expressed as a query over the artifact is cheap; anything that
reaches into the engine is a design smell." Recomputing MOT_FARA from stored
tracklets against a new danger point is a query over the artifact — it is
the report's own headline example of one (§5.1). Duplicating the
stabilization/normalization math here instead of calling the shared function
would violate a more concrete instruction (reuse the same trajectory
substrate STILLA/MOT_FARA already use) for no benefit. See DECISIONS.md B26
for the full reasoning and AGENTS.md's Phase 4 section for the summary.

The recomputed events never touch events/<pass>.jsonl (the frozen engine
output) — review/routes.py merges them in at read time, exactly mirroring
how Phase 3's verdicts overlay the frozen events table without rewriting it.
"""

from __future__ import annotations

from typing import Any

from analysis.events import CATEGORY_MOT_FARA, derive_behavior_events
from analysis.orchestrator import OfflineConfig, OfflineOrchestrator
from analysis.store import ArtifactStore


def _config_from_manifest(store: ArtifactStore) -> OfflineConfig:
    """Reconstruct the run's OfflineConfig from the P5 pass's recorded
    config dict, so the recompute uses the exact same BehaviorAnalyzer
    thresholds as the original run — only the danger point changes.
    Falls back to defaults for any field the manifest doesn't have (older
    sidecars, or fields added after the run was analyzed)."""
    p5_meta = store.manifest.get("passes", {}).get(OfflineOrchestrator.P5_PASS_NAME, {})
    recorded = p5_meta.get("meta", {}).get("config", {})
    defaults = OfflineConfig()
    beh_fields = (
        "beh_window_s",
        "beh_min_history_s",
        "beh_still_speed",
        "beh_still_time_s",
        "beh_toward_speed",
        "beh_toward_angle_deg",
        "beh_toward_time_s",
        "beh_prone_aspect",
    )
    overrides = {f: recorded[f] for f in beh_fields if f in recorded}
    return OfflineConfig(**overrides) if overrides else defaults


def recompute_mot_fara(
    store: ArtifactStore,
    person_by_tracklet: dict[int, int],
    fps: float,
    hazard_x: float,
    hazard_y: float,
) -> list[dict[str, Any]]:
    """Re-derive MOT_FARA events against a manually placed hazard position.

    Reads P2's already-persisted tracklets (the same table the engine's own
    P5 pass reads) and reruns BehaviorAnalyzer's toward-danger logic with a
    *fixed* danger point in frame-pixel space (the same convention the
    overlay canvas already draws in — see review/static/app.js's note that
    artifact boxes are pixel space with no normalization). Pure and
    deterministic given (tracklets, hazard position, config): moving the
    marker back to an already-visited position reproduces byte-identical
    events.

    `frame_w`/`frame_h` aren't needed here — derive_behavior_events only uses
    them (via derive_events) to convert SituationAnalyzer's *normalized*
    fire/smoke position to pixels; a manually placed marker is already given
    in pixel space (the reviewer clicks the overlay canvas directly).

    Returns only the MOT_FARA subset (as plain dicts, `Event.to_dict()`
    shape) — STILLA/IRRATIONELL/HAZARD are unaffected by the danger point
    and stay the engine's original output, merged in by the caller.
    """
    config = _config_from_manifest(store)
    tracklet_rows = list(store.iter_tracklets(OfflineOrchestrator.P2_PASS_NAME))
    events = derive_behavior_events(
        tracklet_rows,
        person_by_tracklet=person_by_tracklet,
        fps=fps,
        frame_w=0,
        frame_h=0,
        config=config,
        danger_px=(hazard_x, hazard_y),
    )
    return [ev.to_dict() for ev in events if ev.category == CATEGORY_MOT_FARA]
