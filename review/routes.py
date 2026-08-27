"""REST API for the review UI.

Three groups of endpoints, mirroring the architecture report's interface rule
("the UI touches only the artifact + annotation REST API, never the engine"):

  Artifact reads (engine output, immutable from the UI's perspective):
    GET  /api/runs                                 — list runs
    GET  /api/runs/{rid}                           — run summary (manifest)
    GET  /api/runs/{rid}/events                    — P5 events log (review state merged in)
    GET  /api/runs/{rid}/events/{eid}              — single event (review state merged in)
    GET  /api/runs/{rid}/tracklets?frame=N         — per-frame boxes for overlay
    GET  /api/runs/{rid}/persons                   — P3 persons
    GET  /api/runs/{rid}/frames/meta?from=N&to=N   — PTS index slice
    GET  /api/runs/{rid}/video                     — original video file (Range)
    GET  /api/runs/{rid}/export?format=csv|json    — event log export

  Annotation writes (human review layer, append-only — report §2.4):
    GET    /api/runs/{rid}/annotations             — live bookmarks, screenshots,
                                                     operator_notes, identity_corrections
                                                     and ground_truth (never toggle-gated)
    POST   /api/runs/{rid}/bookmarks               — add bookmark
    DELETE /api/runs/{rid}/bookmarks/{aid}         — tombstone bookmark
    POST   /api/runs/{rid}/screenshots             — add screenshot (multipart)
    GET    /api/runs/{rid}/screenshots/{aid}/png   — serve stored PNG
    DELETE /api/runs/{rid}/screenshots/{aid}       — tombstone screenshot
    POST   /api/runs/{rid}/events/{eid}/review     — confirm/reject/note an event (Phase 3)

  Evaluation layer (Phase 3 — report §2.4/§5.2-3, the requirement-7 workflow):
    POST   /api/runs/{rid}/operator-notes/import   — parse + store field notes
    GET    /api/runs/{rid}/operator-notes          — list imported notes
    DELETE /api/runs/{rid}/operator-notes/{aid}    — tombstone one imported note
    GET    /api/runs/{rid}/comparison              — AI-vs-operator 3-bucket comparison
    GET    /api/runs/{rid}/debrief                 — standalone HTML debrief export

  Retroactive hazard marker (Phase 4 — report §5.1):
    GET    /api/runs/{rid}/hazard-marker           — current marker (or none)
    POST   /api/runs/{rid}/hazard-marker           — place/move the marker
    DELETE /api/runs/{rid}/hazard-marker           — clear override, revert to engine MOT_FARA
    When a marker is active, GET .../events (and .../comparison, .../debrief)
    transparently serve MOT_FARA recomputed against it (review/hazard.py) —
    events/<pass>.jsonl is never rewritten, exactly like Phase 3's verdicts
    overlay. Existing Phase 3 verdicts on the original MOT_FARA events are
    best-effort carried forward onto the recomputed ones
    (_carry_forward_mot_fara_reviews), keyed by tracklet_id + time proximity.

  Phase 5 (report §5.5-5.9 — each surface behind its own feature toggle,
  see review/config.py; a disabled feature's endpoints answer 404):
    GET    /api/features                            — toggle state for the UI
    GET    /api/runs/{rid}/identity-corrections     — ops + projection status (NOT gated: recorded
                                                      human work stays visible even when the
                                                      dossier toggle is off)
    POST   /api/runs/{rid}/identity-corrections     — record a merge/split op   [FEATURE_DOSSIER]
    DELETE /api/runs/{rid}/identity-corrections/{aid} — undo (tombstone) an op  [FEATURE_DOSSIER]
    GET    /api/runs/{rid}/persons/{pid}/trajectory — downsampled dossier path  [FEATURE_DOSSIER]
    POST   /api/runs/{rid}/ground-truth/import      — import reference truth    [FEATURE_GROUND_TRUTH]
    GET    /api/runs/{rid}/ground-truth             — list reference entries    [FEATURE_GROUND_TRUTH]
    DELETE /api/runs/{rid}/ground-truth/{aid}       — tombstone one entry       [FEATURE_GROUND_TRUTH]
    GET    /api/runs/{rid}/ground-truth/score       — score AI + operator       [FEATURE_GROUND_TRUTH]
    GET    /api/runs/{rid}/compare/{other_rid}      — multi-config run diff     [FEATURE_RUN_COMPARE]
    GET    /api/runs/{rid}/heatmap                  — dwell grid                [FEATURE_HEATMAP]
    (FEATURE_CLIP_EXPORT is fully client-side — the toggle only shows/hides
    the UI controls via /api/features; there is no server endpoint because
    the clip is recorded from the existing composite canvas in the browser,
    keeping the single-annotated-renderer rule from report §2.5.)

    Identity corrections never rewrite persons/<pass>.jsonl: the live ops are
    REPLAYED over the engine's persons at read time
    (review/identity_corrections.py), and the corrected tracklet→person
    mapping flows into /persons, the /tracklets overlay join, served events'
    person_id, and the heatmap's person filter — the same read-time-overlay
    pattern as verdicts (Phase 3) and the hazard marker (Phase 4).

Path traversal guards: run_id and annotation_id are validated strictly
(alphanumeric + dash/underscore) before touching the filesystem — the
review API serves files from a configured directory and a malicious path
must never escape it.
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse

from analysis.events import CATEGORY_MOT_FARA
from analysis.orchestrator import OfflineOrchestrator
from analysis.store import ArtifactStore
from review.annotations import AnnotationStore
from review.comparison import DEFAULT_TOLERANCE_S, compare_events_to_notes
from review.config import ReviewSettings, get_settings
from review.debrief import render_debrief_html
from review.ground_truth import score_ground_truth
from review.hazard import recompute_mot_fara
from review.heatmap import DEFAULT_GRID_W, compute_heatmap
from review.identity_corrections import (
    STATE_MANUAL,
    ProjectionResult,
    apply_corrections,
    merge_overlap_warning,
)
from review.operator_notes import parse_operator_notes
from review.run_compare import DEFAULT_RUN_TOLERANCE_S, RunCompareError, compare_runs

router = APIRouter(prefix="/api")

# Strict id pattern: hex/run-id style only. Rejects slashes, dots, and
# anything that could escape the output directory.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Event ids add a leading category prefix separated by '-' (e.g. "stilla-000001").
_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def _validate_id(value: str, *, pattern: re.Pattern[str] = _ID_RE, label: str = "id") -> str:
    if not pattern.match(value):
        raise HTTPException(status_code=400, detail=f"ogiltigt {label}: {value!r}")
    return value


def _resolve_run(settings: ReviewSettings, run_id: str) -> Path:
    """Resolve <output_dir>/<run_id> and verify a manifest exists."""
    _validate_id(run_id, label="run_id")
    run_dir = (settings.output_dir / run_id).resolve()
    # Defense-in-depth: ensure the resolved path is still under output_dir.
    try:
        run_dir.relative_to(settings.output_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="ogiltigt run_id") from exc
    if not (run_dir / "manifest.json").exists():
        raise HTTPException(status_code=404, detail=f"okänd körning: {run_id}")
    return run_dir


def _open_store(settings: ReviewSettings, run_id: str) -> ArtifactStore:
    return ArtifactStore.open_readonly(_resolve_run(settings, run_id))


def _merge_verdict(event: dict[str, Any], verdict: dict[str, Any] | None) -> dict[str, Any]:
    """Overlay a human verdict onto an engine-persisted event's `review`
    field for API responses. The engine writes events/<pass>.jsonl's
    `review` once (frozen at "unreviewed") and never touches it again —
    confirm/reject/note writes live in the annotations layer's separate
    `verdicts` log instead (see review/annotations.py's module docstring).
    This function is where the two are stitched back together for readers;
    it never writes anything back to the events table."""
    if verdict is None:
        return event
    event = dict(event)
    event["review"] = {
        "state": verdict.get("state", "unreviewed"),
        "note": verdict.get("note"),
        "reviewer": verdict.get("reviewer"),
        "reviewed_at": verdict.get("created_at"),
    }
    return event


def _engine_persons(store: ArtifactStore) -> list[dict[str, Any]]:
    """P3's persisted persons, or [] when P3 didn't run/complete."""
    p3 = OfflineOrchestrator.P3_PASS_NAME
    if store.manifest.get("passes", {}).get(p3, {}).get("status") != "complete":
        return []
    return list(store.iter_persons(p3))


def _tracklet_spans(store: ArtifactStore) -> dict[int, tuple[int, int]]:
    """tracklet_id -> (first_frame, last_frame) from P2. One O(rows) scan —
    only needed when identity corrections exist (split spans + merge
    warnings), so the common no-corrections path never pays for it."""
    spans: dict[int, tuple[int, int]] = {}
    for r in store.iter_tracklets(OfflineOrchestrator.P2_PASS_NAME):
        tid = int(r.get("tracklet_id", -1))
        fno = int(r.get("frame_no", -1))
        cur = spans.get(tid)
        spans[tid] = (fno, fno) if cur is None else (min(cur[0], fno), max(cur[1], fno))
    return spans


def _corrected_projection(
    store: ArtifactStore, ann: AnnotationStore, *, need_spans: bool = True
) -> ProjectionResult:
    """The persons table as served to every reader: engine output with live
    identity-correction ops replayed on top (Phase 5, report §3.3-4).

    Deliberately NOT gated on FEATURE_DOSSIER: the toggle controls the
    dossier UI and new correction writes, but corrections already recorded
    are human review work and keep applying — a config flag must never
    silently un-correct identities (see review/config.py's docstring).

    `need_spans=False` skips the O(rows) tracklets scan that only feeds a
    split's recomputed first/last_seen. The tracklet→person map is identical
    either way, so the per-frame overlay path — which asks for nothing else
    and runs once per displayed frame — must not pay for it.
    """
    persons = _engine_persons(store)
    ops = ann.list_identity_corrections()
    spans = _tracklet_spans(store) if (ops and need_spans) else None
    return apply_corrections(persons, ops, fps=_run_fps(store), tracklet_spans=spans)


def _person_by_tracklet(store: ArtifactStore, ann: AnnotationStore | None = None) -> dict[int, int]:
    """tracklet_id -> person_id (identity corrections applied when `ann` is
    given). Shared by the tracklet overlay endpoint, the hazard-marker
    recompute path, and the heatmap so they never drift on how person_id is
    joined."""
    if ann is not None:
        return _corrected_projection(store, ann, need_spans=False).person_by_tracklet
    person_by_tracklet: dict[int, int] = {}
    for p in _engine_persons(store):
        for tid in p.get("tracklet_ids", []):
            person_by_tracklet[int(tid)] = int(p["person_id"])
    return person_by_tracklet


def _remap_event_persons(events: list[dict[str, Any]], corrected_map: dict[int, int]) -> list[dict[str, Any]]:
    """Re-key served events' person_id through the corrected tracklet→person
    mapping. Engine events recorded person_id at P5 time from P3's original
    clustering; when the reviewer has since split/merged persons, the event
    must follow its tracklet to the corrected identity (evidence.tracklet_id
    is the stable join key — the same reason Phase 4's verdict carry-forward
    keys on it). Events without a tracklet (HAZARD) pass through untouched."""
    out = []
    for ev in events:
        tid = (ev.get("evidence") or {}).get("tracklet_id")
        if tid is not None and int(tid) in corrected_map and ev.get("person_id") is not None:
            new_pid = corrected_map[int(tid)]
            if new_pid != ev.get("person_id"):
                ev = dict(ev)
                ev["person_id"] = new_pid
        out.append(ev)
    return out


def _run_fps(store: ArtifactStore) -> float:
    p1 = OfflineOrchestrator.P1_PASS_NAME
    return store.manifest.get("passes", {}).get(p1, {}).get("meta", {}).get("fps", 25.0)


def _best_review_match(target: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the reviewed original MOT_FARA event that best corresponds to a
    recomputed one: prefer time-overlap, else the closest t_start."""
    overlapping = [
        c for c in candidates if c["t_start"] <= target["t_end"] and c["t_end"] >= target["t_start"]
    ]
    pool = overlapping or candidates
    return min(pool, key=lambda c: abs(c["t_start"] - target["t_start"]))


def _carry_forward_mot_fara_reviews(
    original: list[dict[str, Any]], recomputed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Best-effort carry-forward of Phase 3 verdicts from the engine's
    original MOT_FARA events onto Phase 4's hazard-marker recompute.

    MOT_FARA event ids and spans are not stable across different hazard
    marker positions, so this cannot be an exact identity match — it matches
    by tracklet_id plus closest/overlapping time span, which is the best
    available signal since a tracklet's MOT_FARA span usually only shifts
    slightly when the danger point moves. tracklet_id (not person_id) is the
    matching key because person_id is None on every MOT_FARA event whenever
    P3 didn't run, which would collapse distinct people into one bucket and
    risk carrying one person's verdict onto another's recomputed event.
    Only original events carrying a non-default verdict (state !=
    "unreviewed" or a note) are considered, so an untouched recomputed event
    keeps its default unreviewed review field.
    """
    by_tracklet: dict[Any, list[dict[str, Any]]] = {}
    for ev in original:
        review = ev.get("review") or {}
        if review.get("state", "unreviewed") == "unreviewed" and not review.get("note"):
            continue
        by_tracklet.setdefault(ev.get("evidence", {}).get("tracklet_id"), []).append(ev)
    if not by_tracklet:
        return recomputed
    result = []
    for ev in recomputed:
        candidates = by_tracklet.get(ev.get("evidence", {}).get("tracklet_id"))
        if candidates:
            match = _best_review_match(ev, candidates)
            ev = dict(ev)
            ev["review"] = match["review"]
        result.append(ev)
    return result


def _apply_hazard_override(
    events: list[dict[str, Any]],
    store: ArtifactStore,
    ann: AnnotationStore,
    person_by_tracklet: dict[int, int],
) -> list[dict[str, Any]]:
    """Overlay a retroactive hazard-marker recompute onto the engine's
    MOT_FARA events, if the reviewer has placed a marker (Phase 4, report
    §5.1). Never rewrites events/<pass>.jsonl — this runs at read time only,
    exactly mirroring how _merge_verdict overlays Phase 3's verdicts. When no
    marker is active (never set, or explicitly cleared), returns `events`
    unchanged — the engine's own time-weighted-mean MOT_FARA stands.
    """
    marker = ann.get_hazard_marker()
    if marker is None or marker.get("x") is None:
        return events
    p2 = OfflineOrchestrator.P2_PASS_NAME
    if store.manifest.get("passes", {}).get(p2, {}).get("status") != "complete":
        return events
    recomputed = recompute_mot_fara(
        store,
        person_by_tracklet=person_by_tracklet,
        fps=_run_fps(store),
        hazard_x=None if marker.get("scene_x") is not None else marker["x"],
        hazard_y=None if marker.get("scene_y") is not None else marker["y"],
        hazard_scene=(marker["scene_x"], marker["scene_y"]) if marker.get("scene_x") is not None else None,
        hazard_segment=marker.get("scene_segment"),
    )
    original_mot_fara = [e for e in events if e.get("category") == CATEGORY_MOT_FARA]
    kept = [e for e in events if e.get("category") != CATEGORY_MOT_FARA]
    recomputed = _carry_forward_mot_fara_reviews(original_mot_fara, recomputed)
    merged = kept + recomputed
    merged.sort(key=lambda e: (e["t_start"], e["category"], e["event_id"]))
    return merged


def _resolve_video_path(settings: ReviewSettings, store: ArtifactStore) -> Path | None:
    """Find the source video file for a run.

    The manifest stores the basename (portable across machines); the review
    API resolves it through VIDEO_DIR at serve time. Returns None if the
    file is not present (the UI shows a 'video saknas' empty state)."""
    filename = store.video_filename
    if not filename:
        return None
    candidate = (settings.video_dir / filename).resolve()
    try:
        candidate.relative_to(settings.video_dir.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


# ---- run listing + summary ----


@router.get("/runs")
async def list_runs(settings: ReviewSettings = Depends(get_settings)) -> dict[str, Any]:
    """List analysis runs in OUTPUT_DIR, newest first.

    Each entry includes the manifest's headline fields so the UI can render
    a run picker without N+1 fetches. Runs whose manifest is unreadable are
    skipped silently (a corrupt sidecar shouldn't brick the picker).
    """
    out = settings.output_dir
    if not out.is_dir():
        return {"runs": []}
    runs: list[dict[str, Any]] = []
    for run_dir in out.iterdir():
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            with open(manifest_path) as f:
                m = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        runs.append(
            {
                "run_id": m.get("run_id", run_dir.name),
                "created_at": m.get("created_at", ""),
                "video_filename": m.get("video_filename"),
                "video_hash": m.get("video_hash"),
                "passes": {
                    name: {"status": p.get("status"), "stats": p.get("stats", {})}
                    for name, p in m.get("passes", {}).items()
                },
            }
        )
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return {"runs": runs}


@router.get("/runs/{run_id}")
async def get_run(run_id: str, settings: ReviewSettings = Depends(get_settings)) -> dict[str, Any]:
    """Run summary: manifest + counts + video availability."""
    store = _open_store(settings, run_id)
    m = store.manifest
    video_path = _resolve_video_path(settings, store)
    return {
        "run_id": m.get("run_id", run_id),
        "created_at": m.get("created_at"),
        "video_filename": m.get("video_filename"),
        "video_available": video_path is not None,
        "video_hash": m.get("video_hash"),
        "config_hash": m.get("config_hash"),
        "passes": {
            name: {"status": p.get("status"), "stats": p.get("stats", {})}
            for name, p in m.get("passes", {}).items()
        },
    }


# ---- artifact reads ----


def _merged_events(settings: ReviewSettings, run_id: str) -> tuple[ArtifactStore, list[dict[str, Any]]]:
    """The full event log as served to readers: engine output, with Phase 3
    verdicts and Phase 4's hazard-marker MOT_FARA override both overlaid.
    Shared by get_events/get_event/_load_events_and_notes so the three
    endpoints can never drift on what "the current event log" means."""
    store = _open_store(settings, run_id)
    p5 = OfflineOrchestrator.P5_PASS_NAME
    if store.manifest.get("passes", {}).get(p5, {}).get("status") != "complete":
        raise HTTPException(status_code=409, detail="P5 har inte körts för den här körningen")
    ann = _annotation_store(settings, run_id)
    verdicts = ann.all_verdicts()
    events = [_merge_verdict(ev, verdicts.get(ev["event_id"])) for ev in store.iter_events(p5)]
    # One projection serves both the hazard recompute's person join and the
    # Phase 5 person remap below — both read the map only, never the spans.
    projection = _corrected_projection(store, ann, need_spans=False)
    events = _apply_hazard_override(events, store, ann, projection.person_by_tracklet)
    # Phase 5: follow each event's tracklet to its corrected person identity
    # when split/merge corrections exist (no-op otherwise).
    if projection.changed:
        events = _remap_event_persons(events, projection.person_by_tracklet)
    return store, events


@router.get("/runs/{run_id}/events")
async def get_events(run_id: str, settings: ReviewSettings = Depends(get_settings)) -> dict[str, Any]:
    """Full event log in onset order.

    Each event's `review` field reflects the latest human verdict (Phase 3);
    MOT_FARA reflects the reviewer's hazard marker when one is active
    (Phase 4) — see `_merged_events`."""
    _, events = _merged_events(settings, run_id)
    return {"events": events, "count": len(events)}


@router.get("/runs/{run_id}/events/{event_id}")
async def get_event(
    run_id: str, event_id: str, settings: ReviewSettings = Depends(get_settings)
) -> dict[str, Any]:
    _validate_id(event_id, pattern=_EVENT_ID_RE, label="event_id")
    _, events = _merged_events(settings, run_id)
    for ev in events:
        if ev.get("event_id") == event_id:
            return ev
    raise HTTPException(status_code=404, detail="okänd händelse")


@router.post("/runs/{run_id}/events/{event_id}/review")
async def set_event_review(
    run_id: str,
    event_id: str,
    state: str | None = Form(None, max_length=20),
    note: str | None = Form(None, max_length=4000),
    reviewer: str | None = Form(None, max_length=200),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Confirm/reject/note one event (Phase 3 review queue).

    Any field left out carries forward its previous value (see
    AnnotationStore.set_verdict) — the UI can submit a note-only edit
    without resending the current state, or a state change without
    clobbering an existing note. This writes to the annotations layer's
    `verdicts` log, never to events/<pass>.jsonl."""
    _validate_id(event_id, pattern=_EVENT_ID_RE, label="event_id")
    store = _open_store(settings, run_id)
    p5 = OfflineOrchestrator.P5_PASS_NAME
    if store.manifest.get("passes", {}).get(p5, {}).get("status") != "complete":
        raise HTTPException(status_code=409, detail="P5 har inte körts för den här körningen")
    if not any(ev.get("event_id") == event_id for ev in store.iter_events(p5)):
        raise HTTPException(status_code=404, detail="okänd händelse")
    try:
        return _annotation_store(settings, run_id).set_verdict(
            event_id, state=state, note=note, reviewer=reviewer
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/runs/{run_id}/tracklets")
async def get_tracklets(
    run_id: str,
    frame: int = Query(..., ge=0, description="Frame number to fetch boxes for"),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    """All tracklet boxes present at `frame`, for the overlay canvas.

    Streams P2's tracklets file and filters in-memory — the file is one row
    per (tracklet, frame), so the per-request cost is O(rows) which is fine
    for the review use case (interactive, one frame at a time)."""
    store = _open_store(settings, run_id)
    p2 = OfflineOrchestrator.P2_PASS_NAME
    if store.manifest.get("passes", {}).get(p2, {}).get("status") != "complete":
        raise HTTPException(status_code=409, detail="P2 har inte körts för den här körningen")
    rows = [r for r in store.iter_tracklets(p2) if int(r.get("frame_no", -1)) == frame]
    # Join tracklet → person_id from P3 (if available), with any identity
    # corrections applied, so the overlay shows the same corrected person
    # labels as the rest of the UI.
    person_by_tracklet = _person_by_tracklet(store, _annotation_store(settings, run_id))
    for r in rows:
        tid = int(r.get("tracklet_id", -1))
        r["person_id"] = person_by_tracklet.get(tid)
    return {"frame": frame, "boxes": rows}


@router.get("/runs/{run_id}/tracklets/range")
async def get_tracklets_range(
    run_id: str,
    frame_from: int = Query(..., ge=0, alias="from"),
    frame_to: int = Query(..., ge=0, alias="to"),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    """All tracklet boxes in [frame_from, frame_to], for trail rendering.

    Trails need a window of frames rather than a single frame; this endpoint
    returns them grouped by frame so the client can draw a polyline per
    tracklet without N round-trips."""
    if frame_to < frame_from:
        raise HTTPException(status_code=400, detail="'to' måste vara >= 'from'")
    store = _open_store(settings, run_id)
    p2 = OfflineOrchestrator.P2_PASS_NAME
    if store.manifest.get("passes", {}).get(p2, {}).get("status") != "complete":
        raise HTTPException(status_code=409, detail="P2 har inte körts")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for r in store.iter_tracklets(p2):
        f = int(r.get("frame_no", -1))
        if frame_from <= f <= frame_to:
            grouped.setdefault(f, []).append(r)
    return {"frame_from": frame_from, "frame_to": frame_to, "frames": grouped}


@router.get("/runs/{run_id}/persons")
async def get_persons(run_id: str, settings: ReviewSettings = Depends(get_settings)) -> dict[str, Any]:
    """P3's persons with live identity corrections replayed on top (Phase 5).
    Corrected records carry `corrected: true` + `confirmation_state:
    "manual"` so a consumer can always tell human-corrected identities from
    engine output; ops that no longer resolve are reported in `skipped`
    rather than silently dropped.

    Two counts, deliberately distinct:
      `count`           — every projected person row, transient ones included.
      `unique_count`    — the honest unique-person headline: rows whose
                          confirmation_state is `confirmed` or `manual`
                          (transient excluded). This is NOT the engine's
                          p3.stats.confirmed_persons, which predates the
                          corrections layer and knows nothing of `manual`.
      `transient_count` — `count - unique_count`: short sightings (< the 2s
                          confirm threshold, B16) excluded from the headline.
                          Consumers MUST show it alongside `unique_count` so a
                          bare 0 can never be read as "nobody seen" when
                          someone was briefly observed (the rescue-search case).

    `engine_uncertainty` is deliberately read unchanged from this run's P3
    manifest stats.  Identity corrections change the projected count, but a
    human merge does not prove that P3's recorded blocked near-merge was
    resolved, so projecting/recomputing the band would fabricate evidence.
    """
    store = _open_store(settings, run_id)
    p3 = OfflineOrchestrator.P3_PASS_NAME
    p3_info = store.manifest.get("passes", {}).get(p3, {})
    if p3_info.get("status") != "complete":
        raise HTTPException(status_code=409, detail="P3 har inte körts")
    projection = _corrected_projection(store, _annotation_store(settings, run_id))
    non_transient = ("confirmed", STATE_MANUAL)
    unique_count = sum(1 for p in projection.persons if p.get("confirmation_state") in non_transient)
    return {
        "persons": projection.persons,
        "count": len(projection.persons),
        "unique_count": unique_count,
        # Transient = short sightings (< p3_confirm_s, B16) excluded from
        # unique_count. Surfaced explicitly so a consumer never reads a bare
        # unique_count of 0 as "nobody seen" when someone was briefly observed
        # — the rescue-search danger. count == unique_count + transient_count.
        "transient_count": len(projection.persons) - unique_count,
        "engine_uncertainty": {
            "run_id": store.manifest.get("run_id", run_id),
            "pass": p3,
            "uncertain_merges": p3_info.get("stats", {}).get("uncertain_merges", 0),
        },
        "corrections_applied": len(projection.applied),
        "corrections_skipped": projection.skipped,
    }


@router.get("/runs/{run_id}/frames/meta")
async def get_frames_meta(
    run_id: str,
    frame_from: int = Query(0, ge=0, alias="from"),
    frame_to: int | None = Query(None, ge=0, alias="to"),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Frame metadata (pts_ms) slice, for the overlay's PTS sync.

    The browser's HTMLMediaElement works in media time (seconds); the engine
    works in frame_no. The PTS index bridges them: the overlay walks the
    index to find the frame whose pts_ms is closest to video.currentTime,
    then fetches the matching tracklet boxes."""
    store = _open_store(settings, run_id)
    p1 = OfflineOrchestrator.P1_PASS_NAME
    p1_meta = store.manifest.get("passes", {}).get(p1, {})
    if p1_meta.get("status") != "complete":
        raise HTTPException(status_code=409, detail="P1 har inte körts för den här körningen")
    fps = p1_meta.get("meta", {}).get("fps", 25.0)
    p2_scene: dict[int, dict[str, Any]] = {}
    for rec in store.iter_frames(OfflineOrchestrator.P2_PASS_NAME):
        fn = int(rec.get("frame_no", -1))
        if fn < frame_from:
            continue
        if frame_to is not None and fn > frame_to:
            break
        p2_scene[fn] = rec
    rows: list[dict[str, Any]] = []
    with open(store.run_dir / "frames" / f"{p1}.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fn = int(rec.get("frame_no", -1))
            if fn < frame_from:
                continue
            if frame_to is not None and fn > frame_to:
                break
            row = {"frame_no": fn, "pts_ms": rec.get("pts_ms", fn * 1000.0 / fps)}
            scene = p2_scene.get(fn)
            if scene is not None and scene.get("scene_to_frame") is not None:
                row.update(
                    {
                        "scene_segment": scene.get("scene_segment"),
                        "scene_to_frame": scene.get("scene_to_frame"),
                        "frame_to_scene": scene.get("frame_to_scene"),
                        "scene_confidence": scene.get("scene_confidence"),
                        "scene_linked": scene.get("scene_linked"),
                    }
                )
            rows.append(row)
    return {"frames": rows}


# ---- video serving ----


@router.get("/runs/{run_id}/video", include_in_schema=False)
async def get_video(run_id: str, settings: ReviewSettings = Depends(get_settings)) -> Response:
    """Stream the source video file. FileResponse handles HTTP Range
    requests natively, which the browser needs for seek/scrub on <video>."""
    store = _open_store(settings, run_id)
    video_path = _resolve_video_path(settings, store)
    if video_path is None:
        raise HTTPException(status_code=404, detail="videofilen saknas i VIDEO_DIR")
    return FileResponse(str(video_path), media_type="video/mp4")


# ---- export ----


@router.get("/runs/{run_id}/export")
async def export_events(
    run_id: str,
    format: str = Query("csv", pattern="^(csv|json)$"),
    settings: ReviewSettings = Depends(get_settings),
) -> Response:
    """Export the AI event log (CSV for spreadsheets, JSON for round-trip).

    Per the architecture report, exports bundle AI output only — annotation
    export (bookmarks/screenshots) is a separate concern handled through the
    annotation endpoints, since annotations are human-authored and live in
    a different table with different persistence semantics."""
    store = _open_store(settings, run_id)
    p5 = OfflineOrchestrator.P5_PASS_NAME
    if store.manifest.get("passes", {}).get(p5, {}).get("status") != "complete":
        raise HTTPException(status_code=409, detail="P5 har inte körts")
    events = list(store.iter_events(p5))

    if format == "json":
        payload = {"run_id": run_id, "events": events}
        return JSONResponse(
            payload,
            headers={"Content-Disposition": f'attachment; filename="{run_id}_events.json"'},
        )

    # CSV: flat header row, evidence + review serialized as JSON strings.
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["event_id", "category", "person_id", "t_start", "t_end", "confidence", "evidence", "review"]
    )
    for ev in events:
        writer.writerow(
            [
                ev.get("event_id"),
                ev.get("category"),
                ev.get("person_id") if ev.get("person_id") is not None else "",
                ev.get("t_start"),
                ev.get("t_end"),
                ev.get("confidence"),
                json.dumps(ev.get("evidence", {}), ensure_ascii=False),
                json.dumps(ev.get("review", {}), ensure_ascii=False),
            ]
        )
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_events.csv"'},
    )


# ---- annotations: bookmarks ----


def _annotation_store(settings: ReviewSettings, run_id: str) -> AnnotationStore:
    """Construct an AnnotationStore for the run, after validating run_id."""
    run_dir = _resolve_run(settings, run_id)
    return AnnotationStore(run_dir)


@router.get("/runs/{run_id}/annotations")
async def get_annotations(run_id: str, settings: ReviewSettings = Depends(get_settings)) -> dict[str, Any]:
    """All live entity-per-row annotations for the run, grouped by kind:
    bookmarks, screenshots, operator_notes, identity_corrections and
    ground_truth. Every kind stays readable here regardless of feature
    toggles (see review/annotations.py's module docstring). Verdicts and the
    hazard marker are latest-row-wins values rather than entities and are
    served through their own endpoints (`.../events`, `.../hazard-marker`)."""
    return _annotation_store(settings, run_id).all_annotations()


@router.post("/runs/{run_id}/bookmarks", status_code=201)
async def add_bookmark(
    run_id: str,
    t: float = Form(..., ge=0),
    label: str = Form(..., min_length=1, max_length=200),
    note: str | None = Form(None, max_length=4000),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    return _annotation_store(settings, run_id).add_bookmark(t=t, label=label, note=note)


@router.delete("/runs/{run_id}/bookmarks/{aid}", status_code=200)
async def delete_bookmark(
    run_id: str, aid: str, settings: ReviewSettings = Depends(get_settings)
) -> dict[str, Any]:
    _validate_id(aid, label="annotation_id")
    ok = _annotation_store(settings, run_id).delete_bookmark(aid)
    if not ok:
        raise HTTPException(status_code=404, detail="bokmärket finns inte")
    return {"deleted": aid}


# ---- annotations: screenshots ----


@router.post("/runs/{run_id}/screenshots", status_code=201)
async def add_screenshot(
    run_id: str,
    t: float = Form(..., ge=0),
    label: str = Form(..., min_length=1, max_length=200),
    note: str | None = Form(None, max_length=4000),
    png: bytes | None = File(None, description="Client-composited PNG (optional)"),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Record a screenshot taken at video time t.

    The PNG is composited in the browser (video frame + overlay canvas) and
    uploaded as-is; the server never renders a frame. Per report §2.5 this
    retires the dual-renderer hazard (snapshot.py's server-side renderer).
    The PNG is optional — a metadata-only row still anchors the timestamp
    in the review log."""
    return _annotation_store(settings, run_id).add_screenshot(t=t, label=label, note=note, png_bytes=png)


@router.get("/runs/{run_id}/screenshots/{aid}/png", include_in_schema=False)
async def get_screenshot_png(
    run_id: str, aid: str, settings: ReviewSettings = Depends(get_settings)
) -> Response:
    _validate_id(aid, label="annotation_id")
    path = _annotation_store(settings, run_id).screenshot_png_path(aid)
    if path is None:
        raise HTTPException(status_code=404, detail="PNG saknas för skärmdumpen")
    return FileResponse(str(path), media_type="image/png")


@router.delete("/runs/{run_id}/screenshots/{aid}", status_code=200)
async def delete_screenshot(
    run_id: str, aid: str, settings: ReviewSettings = Depends(get_settings)
) -> dict[str, Any]:
    _validate_id(aid, label="annotation_id")
    ok = _annotation_store(settings, run_id).delete_screenshot(aid)
    if not ok:
        raise HTTPException(status_code=404, detail="skärmdumpen finns inte")
    return {"deleted": aid}


# ---- Phase 3: operator-notes import ----


@router.post("/runs/{run_id}/operator-notes/import", status_code=201)
async def import_operator_notes(
    run_id: str,
    text: str = Form(..., min_length=1, max_length=200_000),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Parse and store a blob of operator field notes.

    See review/operator_notes.py for the accepted (deliberately forgiving)
    format. Best-effort: unparseable lines are reported as warnings rather
    than failing the whole import — one garbled line, transcribed under
    time pressure, must never lose the rest of a field report."""
    _resolve_run(settings, run_id)
    result = parse_operator_notes(text)
    ann = _annotation_store(settings, run_id)
    imported = [ann.add_operator_note(t=n.t, text=n.text, raw_line=n.raw_line) for n in result.notes]
    return {
        "imported": imported,
        "warnings": [
            {"line": w.line_no, "raw_line": w.raw_line, "reason": w.reason} for w in result.warnings
        ],
    }


@router.get("/runs/{run_id}/operator-notes")
async def get_operator_notes(run_id: str, settings: ReviewSettings = Depends(get_settings)) -> dict[str, Any]:
    notes = _annotation_store(settings, run_id).list_operator_notes()
    return {"notes": notes, "count": len(notes)}


@router.delete("/runs/{run_id}/operator-notes/{aid}", status_code=200)
async def delete_operator_note(
    run_id: str, aid: str, settings: ReviewSettings = Depends(get_settings)
) -> dict[str, Any]:
    _validate_id(aid, label="annotation_id")
    ok = _annotation_store(settings, run_id).delete_operator_note(aid)
    if not ok:
        raise HTTPException(status_code=404, detail="observationen finns inte")
    return {"deleted": aid}


# ---- Phase 3: AI-vs-operator comparison + HTML debrief ----


def _load_events_and_notes(
    settings: ReviewSettings, run_id: str
) -> tuple[ArtifactStore, list[dict[str, Any]], list[dict[str, Any]]]:
    """Shared read path for /comparison and /debrief: the merged event log
    (verdicts + hazard-marker override, see _merged_events) + live imported
    operator notes."""
    store, events = _merged_events(settings, run_id)
    notes = _annotation_store(settings, run_id).list_operator_notes()
    return store, events, notes


@router.get("/runs/{run_id}/comparison")
async def get_comparison(
    run_id: str,
    tolerance_s: float = Query(DEFAULT_TOLERANCE_S, ge=0),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Three-bucket AI-vs-operator comparison (report §5.3): found by both /
    AI-only / operator-only, with a signed time-to-detection delta on
    matched pairs. Computed fresh on every call from the live events +
    operator_notes — see review/comparison.py for the matching rules and
    why the default tolerance is 60s."""
    _, events, notes = _load_events_and_notes(settings, run_id)
    result = compare_events_to_notes(events, notes, tolerance_s=tolerance_s)
    return {
        "tolerance_s": result.tolerance_s,
        "counts": result.counts,
        "both": [{"event": m.event, "note": m.note, "delta_s": m.delta_s} for m in result.both],
        "ai_only": result.ai_only,
        "operator_only": result.operator_only,
    }


@router.get("/runs/{run_id}/debrief", include_in_schema=False)
async def get_debrief(
    run_id: str,
    tolerance_s: float = Query(DEFAULT_TOLERANCE_S, ge=0),
    settings: ReviewSettings = Depends(get_settings),
) -> Response:
    """Standalone HTML training-debrief export (report §5.3's stated point
    of the operator-comparison feature) — self-contained, no server needed
    to view it. See review/debrief.py."""
    store, events, notes = _load_events_and_notes(settings, run_id)
    result = compare_events_to_notes(events, notes, tolerance_s=tolerance_s)
    html = render_debrief_html(
        run_id,
        result,
        generated_at=datetime.now(timezone.utc).isoformat(),
        video_filename=store.manifest.get("video_filename"),
    )
    return Response(
        content=html,
        media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_debrief.html"'},
    )


# ---- Phase 4: retroactive hazard marker ----


@router.get("/runs/{run_id}/hazard-marker")
async def get_hazard_marker(run_id: str, settings: ReviewSettings = Depends(get_settings)) -> dict[str, Any]:
    """Current hazard marker, or `{"active": false}` if none is set (never
    placed, or explicitly cleared) — the engine's own detected danger point
    is in effect in that case."""
    marker = _annotation_store(settings, run_id).get_hazard_marker()
    if marker is None or marker.get("x") is None:
        return {"active": False}
    payload = {"active": True, "x": marker["x"], "y": marker["y"], "note": marker.get("note")}
    if marker.get("scene_x") is not None:
        payload.update(
            {
                "coordinate_space": "scene",
                "anchor_frame": marker.get("anchor_frame"),
                "scene_x": marker.get("scene_x"),
                "scene_y": marker.get("scene_y"),
                "scene_segment": marker.get("scene_segment"),
                "scene_confidence": marker.get("scene_confidence"),
            }
        )
    return payload


@router.post("/runs/{run_id}/hazard-marker", status_code=201)
async def set_hazard_marker(
    run_id: str,
    x: float = Form(..., ge=0),
    y: float = Form(..., ge=0),
    frame_no: int | None = Form(None, ge=0),
    note: str | None = Form(None, max_length=4000),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Place or move the hazard marker (report §5.1, DECISIONS B29).

    `x`/`y` are intrinsic frame pixels from the overlay canvas. For a new
    sidecar, `frame_no` selects P2's inverse transform and the server stores
    the corresponding local scene point as provenance. Older sidecars without
    scene transforms retain fixed-pixel behavior. The next GET .../events
    immediately recomputes MOT_FARA against the marker.
    """
    store = _open_store(settings, run_id)
    scene = None
    scene_supported = False
    for row in store.iter_frames(OfflineOrchestrator.P2_PASS_NAME):
        if row.get("frame_to_scene") is not None:
            scene_supported = True
        if frame_no is not None and int(row.get("frame_no", -1)) == frame_no:
            scene = row
            if row.get("frame_to_scene") is not None:
                break
    if scene_supported and (scene is None or scene.get("frame_to_scene") is None):
        raise HTTPException(status_code=422, detail="faromarkören kräver en giltig ankarruta")
    scene_kwargs: dict[str, Any] = {}
    if scene is not None and scene.get("frame_to_scene") is not None:
        from analysis.scene import transform_point

        scene_x, scene_y = transform_point(scene["frame_to_scene"], (x, y))
        scene_kwargs = {
            "anchor_frame": frame_no,
            "scene_x": scene_x,
            "scene_y": scene_y,
            "scene_segment": int(scene["scene_segment"]),
            "scene_confidence": float(scene.get("scene_confidence") or 0.0),
        }
    return _annotation_store(settings, run_id).set_hazard_marker(x=x, y=y, note=note, **scene_kwargs)


@router.delete("/runs/{run_id}/hazard-marker", status_code=200)
async def delete_hazard_marker(
    run_id: str, settings: ReviewSettings = Depends(get_settings)
) -> dict[str, Any]:
    """Clear the manual override — MOT_FARA reverts to the engine's own
    time-weighted-mean danger point on the next read."""
    _annotation_store(settings, run_id).clear_hazard_marker()
    return {"active": False}


# ---- Phase 5: feature toggles ----


def _require_feature(settings: ReviewSettings, attr: str, env_name: str) -> None:
    """Gate a Phase 5 endpoint on its toggle. 404 (not 403): a disabled
    feature should look absent, and the detail names the env var so an
    operator immediately knows which switch to flip."""
    if not getattr(settings, attr):
        raise HTTPException(status_code=404, detail=f"funktionen är avstängd ({env_name}=0)")


@router.get("/features")
async def get_features(settings: ReviewSettings = Depends(get_settings)) -> dict[str, bool]:
    """Phase 5 toggle state — the UI hides tabs/buttons for disabled features."""
    return settings.features_payload()


# ---- Phase 5: identity corrections (dossier split/merge) ----


@router.get("/runs/{run_id}/identity-corrections")
async def get_identity_corrections(
    run_id: str, settings: ReviewSettings = Depends(get_settings)
) -> dict[str, Any]:
    """Live correction ops + how the current projection treated them.

    Deliberately NOT gated on FEATURE_DOSSIER: recorded corrections keep
    applying to read-time projections when the toggle is off (see
    review/config.py), so their existence must stay inspectable — a config
    flag never hides recorded human work."""
    store = _open_store(settings, run_id)
    ann = _annotation_store(settings, run_id)
    projection = _corrected_projection(store, ann, need_spans=False)
    return {
        "corrections": ann.list_identity_corrections(),
        "applied": projection.applied,
        "skipped": projection.skipped,
    }


@router.post("/runs/{run_id}/identity-corrections", status_code=201)
async def add_identity_correction(
    run_id: str,
    op: str = Form(..., max_length=10),
    person_ids: str | None = Form(None, max_length=1000, description="merge: kommaseparerade person-id"),
    person_id: int | None = Form(None, ge=0),
    tracklet_ids: str | None = Form(None, max_length=4000, description="split: kommaseparerade tracklet-id"),
    reason: str | None = Form(None, max_length=4000),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Record one manual split/merge correction (Phase 5, report §3.3-4).

    The op is keyed by the TRACKLET SETS of the persons it names, captured
    here from the current projection — person_id is positional in P3's
    output and would silently resolve to a different person after a
    re-analysis (see review/identity_corrections.py). The submitted ids are
    still stored, as provenance, together with the run's video_hash and
    config_hash so a later skip is diagnosable.

    Validated by DRY-RUNNING the op on top of the current projection (engine
    persons + every earlier live op): an op that wouldn't apply is rejected
    with the projection's own skip reason instead of being persisted as a
    dead row. A merge of persons visible simultaneously is allowed — the
    human may know better than P3's hard gate (e.g. tracker twin boxes) —
    but answers with a warning naming the contradiction.
    """
    _require_feature(settings, "feature_dossier", "FEATURE_DOSSIER")
    store = _open_store(settings, run_id)
    p3 = OfflineOrchestrator.P3_PASS_NAME
    if store.manifest.get("passes", {}).get(p3, {}).get("status") != "complete":
        raise HTTPException(status_code=409, detail="P3 har inte körts — inga personer att korrigera")

    def parse_ids(raw: str | None) -> list[int]:
        try:
            return [int(x) for x in (raw or "").replace(";", ",").split(",") if x.strip()]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"ogiltig id-lista: {raw!r}") from exc

    if op not in ("merge", "split"):
        raise HTTPException(status_code=422, detail=f"okänd operation: {op!r}")

    ann = _annotation_store(settings, run_id)
    ops = ann.list_identity_corrections()
    spans = _tracklet_spans(store)
    engine_persons = _engine_persons(store)
    # The state the correction acts on: engine persons + every earlier live
    # op. Both the tracklet-set keys and the overlap warning read from it.
    before = apply_corrections(engine_persons, ops, fps=_run_fps(store), tracklet_spans=spans)
    projected = {int(p["person_id"]): p for p in before.persons}

    def tracklet_key(pid: int | None) -> list[int]:
        rec = projected.get(int(pid)) if pid is not None else None
        if rec is None:
            raise HTTPException(status_code=422, detail=f"person {pid} saknas i aktuell projektion")
        return sorted(int(t) for t in rec.get("tracklet_ids", []))

    candidate: dict[str, Any] = {"annotation_id": "__candidate__", "op": op, "reason": reason}
    if op == "merge":
        merge_ids = parse_ids(person_ids)
        if len(set(merge_ids)) < 2:
            raise HTTPException(status_code=422, detail="sammanslagning kräver minst två olika person-id")
        candidate["person_ids"] = merge_ids
        candidate["member_tracklet_ids"] = [tracklet_key(p) for p in merge_ids]
    else:
        if person_id is None:
            raise HTTPException(status_code=422, detail="delning kräver ett person-id")
        candidate["person_id"] = person_id
        candidate["tracklet_ids"] = parse_ids(tracklet_ids)
        candidate["source_tracklet_ids"] = tracklet_key(person_id)

    dry = apply_corrections(engine_persons, ops + [candidate], fps=_run_fps(store), tracklet_spans=spans)
    rejected = next((s for s in dry.skipped if s["annotation_id"] == "__candidate__"), None)
    if rejected is not None:
        raise HTTPException(status_code=422, detail=rejected["reason"])

    warning = None
    if op == "merge":
        warning = merge_overlap_warning(projected, candidate["person_ids"], spans)

    row = ann.add_identity_correction(
        op=op,
        person_ids=candidate.get("person_ids"),
        person_id=candidate.get("person_id"),
        tracklet_ids=candidate.get("tracklet_ids"),
        member_tracklet_ids=candidate.get("member_tracklet_ids"),
        source_tracklet_ids=candidate.get("source_tracklet_ids"),
        reason=reason,
        video_hash=store.manifest.get("video_hash"),
        config_hash=store.manifest.get("config_hash"),
    )
    return {"correction": row, "warning": warning}


@router.delete("/runs/{run_id}/identity-corrections/{aid}", status_code=200)
async def delete_identity_correction(
    run_id: str, aid: str, settings: ReviewSettings = Depends(get_settings)
) -> dict[str, Any]:
    """Undo a correction (tombstone). Later ops that depended on it are
    skipped by the projection and reported via GET, never guessed at."""
    _require_feature(settings, "feature_dossier", "FEATURE_DOSSIER")
    _validate_id(aid, label="annotation_id")
    ok = _annotation_store(settings, run_id).delete_identity_correction(aid)
    if not ok:
        raise HTTPException(status_code=404, detail="korrigeringen finns inte")
    return {"deleted": aid}


@router.get("/runs/{run_id}/persons/{person_id}/trajectory")
async def get_person_trajectory(
    run_id: str,
    person_id: int,
    max_points: int = Query(1500, ge=10, le=10000),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Downsampled foot-center path for one corrected person.

    New sidecars use P2's local scene coordinates and expose segment ids so
    the client never joins paths across an unlinked camera cut.  Older
    sidecars retain their raw-frame fallback.  Stride-downsampling keeps a
    film-long trajectory bounded.
    """
    _require_feature(settings, "feature_dossier", "FEATURE_DOSSIER")
    store = _open_store(settings, run_id)
    p2 = OfflineOrchestrator.P2_PASS_NAME
    if store.manifest.get("passes", {}).get(p2, {}).get("status") != "complete":
        raise HTTPException(status_code=409, detail="P2 har inte körts")
    projection = _corrected_projection(store, _annotation_store(settings, run_id), need_spans=False)
    rec = next((p for p in projection.persons if int(p["person_id"]) == person_id), None)
    if rec is None:
        raise HTTPException(status_code=404, detail="okänd person")
    tids = {int(t) for t in rec.get("tracklet_ids", [])}
    fps = _run_fps(store)
    rows = [r for r in store.iter_tracklets(p2) if int(r.get("tracklet_id", -1)) in tids]
    scene_mode = bool(rows) and all(r.get("scene_pos") is not None for r in rows)
    pts = []
    for r in rows:
        x0, y0, x1, y1 = (float(v) for v in r["xyxy"])
        fno = int(r["frame_no"])
        point = {
            "frame_no": fno,
            "t": round(fno / fps, 3),
            "x": round(float(r["scene_pos"][0]) if scene_mode else (x0 + x1) / 2.0, 1),
            "y": round(float(r["scene_pos"][1]) if scene_mode else y1, 1),
            "tracklet_id": int(r["tracklet_id"]),
        }
        if scene_mode:
            point["scene_segment"] = int(r["scene_segment"])
        pts.append(point)
    pts.sort(key=lambda p: (p["frame_no"], p["tracklet_id"]))
    n_total = len(pts)
    stride = max(1, -(-n_total // max_points))  # ceil division
    return {
        "person_id": person_id,
        "coordinate_space": "scene" if scene_mode else "frame",
        "n_total": n_total,
        "stride": stride,
        "points": pts[::stride],
    }


# ---- Phase 5: ground truth ("facit") ----


@router.post("/runs/{run_id}/ground-truth/import", status_code=201)
async def import_ground_truth(
    run_id: str,
    text: str = Form(..., min_length=1, max_length=200_000),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Parse and store the exercise leader's reference truth. Same forgiving
    timed-line format as operator notes (one parser, one syntax to learn —
    see review/ground_truth.py); unparseable lines become warnings."""
    _require_feature(settings, "feature_ground_truth", "FEATURE_GROUND_TRUTH")
    _resolve_run(settings, run_id)
    result = parse_operator_notes(text)
    ann = _annotation_store(settings, run_id)
    imported = [ann.add_ground_truth(t=n.t, text=n.text, raw_line=n.raw_line) for n in result.notes]
    return {
        "imported": imported,
        "warnings": [
            {"line": w.line_no, "raw_line": w.raw_line, "reason": w.reason} for w in result.warnings
        ],
    }


@router.get("/runs/{run_id}/ground-truth")
async def get_ground_truth(run_id: str, settings: ReviewSettings = Depends(get_settings)) -> dict[str, Any]:
    _require_feature(settings, "feature_ground_truth", "FEATURE_GROUND_TRUTH")
    rows = _annotation_store(settings, run_id).list_ground_truth()
    return {"entries": rows, "count": len(rows)}


@router.delete("/runs/{run_id}/ground-truth/{aid}", status_code=200)
async def delete_ground_truth(
    run_id: str, aid: str, settings: ReviewSettings = Depends(get_settings)
) -> dict[str, Any]:
    _require_feature(settings, "feature_ground_truth", "FEATURE_GROUND_TRUTH")
    _validate_id(aid, label="annotation_id")
    ok = _annotation_store(settings, run_id).delete_ground_truth(aid)
    if not ok:
        raise HTTPException(status_code=404, detail="facitposten finns inte")
    return {"deleted": aid}


@router.get("/runs/{run_id}/ground-truth/score")
async def get_ground_truth_score(
    run_id: str,
    tolerance_s: float = Query(DEFAULT_TOLERANCE_S, ge=0),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Score BOTH the AI's events and the operator's notes against the
    reference truth (report §5.6). Recomputed fresh on every call from live
    data (like /comparison) — reproducible by construction. Events include
    the verdict/hazard/correction overlays, so the score reflects what the
    reviewer actually sees."""
    _require_feature(settings, "feature_ground_truth", "FEATURE_GROUND_TRUTH")
    _, events = _merged_events(settings, run_id)
    ann = _annotation_store(settings, run_id)
    score = score_ground_truth(
        ann.list_ground_truth(), events, ann.list_operator_notes(), tolerance_s=tolerance_s
    )
    return {
        "tolerance_s": score.tolerance_s,
        "counts": score.counts,
        "entries": score.entries,
        "ai_unmatched": score.ai_unmatched,
        "operator_unmatched": score.operator_unmatched,
    }


# ---- Phase 5: multi-config run comparison ----


@router.get("/runs/{run_id}/compare/{other_run_id}")
async def get_run_comparison(
    run_id: str,
    other_run_id: str,
    tolerance_s: float = Query(DEFAULT_RUN_TOLERANCE_S, ge=0),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Diff two completed runs of the same video (report §5.7). The diff is
    cheap; producing the runs is the expensive batch part and always happens
    via the analyze CLI, never from here — see review/run_compare.py."""
    _require_feature(settings, "feature_run_compare", "FEATURE_RUN_COMPARE")
    store_a = _open_store(settings, run_id)
    store_b = _open_store(settings, other_run_id)
    try:
        return compare_runs(store_a, store_b, tolerance_s=tolerance_s)
    except RunCompareError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


# ---- Phase 5: dwell/coverage heatmap ----


@router.get("/runs/{run_id}/heatmap")
async def get_heatmap(
    run_id: str,
    person_id: int | None = Query(None, ge=0),
    grid: int = Query(DEFAULT_GRID_W, ge=8, le=160),
    settings: ReviewSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Dwell grid over P2's tracklets (report §5.9), optionally filtered to
    one (corrected) person for the dossier. Frame dims come from P1's pass
    meta; older sidecars lack them and fall back to the tracklet boxes'
    extent (slightly tight, harmless for a density overlay)."""
    _require_feature(settings, "feature_heatmap", "FEATURE_HEATMAP")
    store = _open_store(settings, run_id)
    p2 = OfflineOrchestrator.P2_PASS_NAME
    if store.manifest.get("passes", {}).get(p2, {}).get("status") != "complete":
        raise HTTPException(status_code=409, detail="P2 har inte körts")
    p1_meta = store.manifest.get("passes", {}).get(OfflineOrchestrator.P1_PASS_NAME, {}).get("meta", {})
    frame_w = int(p1_meta.get("width") or 0)
    frame_h = int(p1_meta.get("height") or 0)
    if frame_w <= 0 or frame_h <= 0:
        rows = list(store.iter_tracklets(p2))
        frame_w = int(max((float(r["xyxy"][2]) for r in rows), default=0.0)) + 1
        frame_h = int(max((float(r["xyxy"][3]) for r in rows), default=0.0)) + 1
    else:
        rows = store.iter_tracklets(p2)
    pmap = _person_by_tracklet(store, _annotation_store(settings, run_id)) if person_id is not None else None
    return compute_heatmap(
        rows,
        fps=_run_fps(store),
        frame_w=frame_w,
        frame_h=frame_h,
        grid_w=grid,
        person_id=person_id,
        person_by_tracklet=pmap,
    )
