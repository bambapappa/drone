#!/usr/bin/env python3
"""Offline drone video analysis CLI.

Usage:
    analyze <video> [--output DIR] [--model PATH] [--device DEVICE]
                    [--imgsz N] [--detect-conf F] [--display-conf F]
                    [--tiles N] [--seed N] [--resume RUN_ID | --resume latest]
                    [--reid-weights PATH] [--reid-floor N]
                    [--no-p3] [--no-p5]

Runs ingest + P1 (detection) + P2 (tracking) + P3 (identity) + P5 (events)
end-to-end. The analysis sidecar is written to <output>/<run_id>/ with
manifest.json, frames/, detections/, tracklets/, persons/, events/,
annotations/, and checkpoints/.

P1 is stateless and checkpointed/resumable: by default every invocation
mints a fresh run_id and a fresh sidecar — re-running the same command does
NOT resume automatically. To continue an interrupted P1, pass --resume
<run_id> (or --resume latest to resolve the most recent matching run under
--output). Resume refuses to continue unless the target run's video hash,
config hash, code version, and tracker library version all match the
current invocation, so it can never silently splice together a run across a
config, code, or tracker-library change.

P2/P3/P5 always re-run in full (cheap and deterministic given the same P1
output) — they are never checkpointed and --resume does not affect them.

P3 (identity) re-associates P2's tracklets into persons via global
agglomerative clustering gated by temporal-overlap exclusion, spatio-
temporal plausibility, and appearance similarity. The reported unique-person
count is honestly uncertainty-banded ("N unika, varav M osäkra
sammanslagningar") — same-clothing confusion is a fundamental appearance-
method limit (DECISIONS B4) that the tool surfaces rather than hides.

P5 (events) replays the carried-over BehaviorAnalyzer + SituationAnalyzer
over the persisted tracklets + frame stream and diffs their per-frame status
into discrete STILLA / MOT_FARA / HAZARD events with onset/offset timestamps
and confidence, keyed to person_id when P3 ran. IRRATIONELL is Phase 4.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Allow running as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Offline drone video analysis — batch ingest + P1 detection.",
        epilog="For the full architecture and phase plan, see the project report.",
    )
    ap.add_argument("video", help="Path to the video file to analyze")
    ap.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for the sidecar store (default: <video_dir>_analysis)",
    )
    ap.add_argument("--model", default=None, help="YOLO model path (default: yolo11n.pt)")
    ap.add_argument("--device", default="cpu", help="Torch device (cpu, mps, cuda:0)")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument(
        "--detect-conf",
        type=float,
        default=0.08,
        help="Inference-time confidence cutoff for what P1 persists (matches BoT-SORT's "
        "track_low_thresh so P2's low-score association bucket has data). Not a display filter.",
    )
    ap.add_argument(
        "--display-conf",
        type=float,
        default=0.30,
        help="Downstream display/analysis confidence threshold, recorded in the manifest "
        "for provenance. Applied by consumers, not at inference time.",
    )
    ap.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    ap.add_argument("--tiles", type=int, default=1, help="NxN tiled inference (1=off)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    ap.add_argument(
        "--track-buffer-s",
        type=float,
        default=8.0,
        help="Track buffer in video seconds (BoT-SORT track_buffer = fps × this)",
    )
    ap.add_argument(
        "--reid-weights",
        default=None,
        help="Path to a TorchScript ReID model (OSNet-class) for per-detection "
        "appearance embeddings. Absent → HSV-only embeddings (weaker but "
        "functional); present → OSNet primary with HSV fallback below the crop "
        "floor. Recorded in the manifest for provenance.",
    )
    ap.add_argument(
        "--reid-floor",
        type=int,
        default=32,
        help="Min crop side (px) below which the HSV fallback fires for the ReID "
        "embedder (10 px people at altitude are below any ReID model's input).",
    )
    ap.add_argument(
        "--no-p3",
        action="store_true",
        help="Stop after P1+P2 (detection+tracking) without running the P3 "
        "identity pass. By default P3 runs and the honest unique-person count "
        "is reported. Implies --no-p5 (P5 needs P3's tracklet→person map for "
        "person-keyed events).",
    )
    ap.add_argument(
        "--no-p5",
        action="store_true",
        help="Skip the P5 event-derivation pass. By default P5 runs after P3 "
        "and writes events/ for the review UI.",
    )
    ap.add_argument(
        "--resume",
        default=None,
        metavar="RUN_ID",
        help="Resume an existing run (by run_id, or 'latest' to resolve the most "
        "recent matching run under --output). Default: mint a fresh run_id.",
    )

    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or f"{video_path.stem}_analysis"
    output_dir = str(Path(output_dir).resolve())

    print(f"Analyze:          {video_path.name}")
    print(f"Output:           {output_dir}")
    print(f"Model:            {args.model or 'yolo11n.pt'}")
    print(f"Device:           {args.device}")
    print(f"Resolution:       {args.imgsz}")
    print(f"Detect/Display conf/IoU: {args.detect_conf}/{args.display_conf}/{args.iou}")
    print(f"Tiles:            {args.tiles}")
    print(f"Seed:             {args.seed}")
    print(f"Track buffer:     {args.track_buffer_s}s")
    print()

    # ---- Ingest ----
    print("--- Ingest ---")
    t0 = time.monotonic()

    from analysis.ingest import ingest as do_ingest

    meta, frame_store = do_ingest(str(video_path))
    frame_store.close()

    print(f"  File:           {video_path.name}")
    print(f"  Resolution:     {meta.width}x{meta.height}")
    print(f"  FPS:            {meta.fps:.1f}")
    print(f"  Total frames:   {meta.total_frames}")
    print(f"  Duration:       {meta.total_frames / meta.fps:.1f}s")
    print(f"  Video hash:     {meta.video_hash}")
    print(f"  PiP layout:     {meta.pip_layout or 'none'}")
    if meta.pip_region:
        print(f"  PiP region:     {meta.pip_region}")
    if meta.active_roi:
        print(f"  Active ROI:     {meta.active_roi}")

    ingest_elapsed = time.monotonic() - t0
    print(f"  Ingest time:    {ingest_elapsed:.1f}s")
    print()

    # ---- Config ----
    from analysis.orchestrator import OfflineConfig
    from analysis.store import ArtifactStore

    config = OfflineConfig(
        model=args.model or os.environ.get("MODEL", "yolo11n.pt"),
        device=args.device,
        imgsz=args.imgsz,
        detect_conf=args.detect_conf,
        display_conf=args.display_conf,
        iou=args.iou,
        tiles=args.tiles,
        seed=args.seed,
        track_buffer_s=args.track_buffer_s,
        reid_weights=args.reid_weights,
        reid_floor=args.reid_floor,
    )

    config_hash = ArtifactStore.config_hash_from_settings(config.to_dict())

    # ---- Store ----
    from analysis.store import ResumeValidationError

    if args.resume:
        resume_id = args.resume
        if resume_id == "latest":
            resume_id = ArtifactStore.resolve_latest(output_dir, meta.video_hash, config_hash)
            if resume_id is None:
                print(
                    f"Error: no existing run under {output_dir} matches this "
                    "video and config to resume from.",
                    file=sys.stderr,
                )
                sys.exit(1)
        try:
            store = ArtifactStore.open_existing(output_dir, resume_id, meta.video_hash, config_hash)
        except ResumeValidationError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Resuming run:     {store.run_id}")
    else:
        store = ArtifactStore(
            output_dir=output_dir,
            video_hash=meta.video_hash,
            config_hash=config_hash,
        )
        store.create()
    # Record the source video basename so the review UI can locate the file
    # through VIDEO_DIR. Stored as basename only — portable across machines.
    store.set_video_filename(video_path.name)
    print(f"Sidecar store:    {store.run_dir}")
    print(f"  Run ID:         {store.run_id}")
    print()

    # ---- P1 Detection Pass ----
    print("--- P1 Detection ---")
    from analysis.orchestrator import OfflineOrchestrator

    orchestrator = OfflineOrchestrator(meta, store, config)
    orchestrator.run_pass_p1()

    pass_info = store._manifest.get("passes", {}).get(OfflineOrchestrator.P1_PASS_NAME, {})
    stats = pass_info.get("stats", {})
    print(f"  Frames:         {stats.get('frames_processed', '?')}/{meta.total_frames}")
    print(f"  Detections:     {stats.get('total_detections', '?')}")
    print(f"  Time:           {stats.get('elapsed_s', '?')}s")
    eff = stats.get("fps_effective", 0)
    print(f"  Effective fps:  {eff}")
    print()

    p1_status = pass_info.get("status")
    if p1_status != "complete":
        print(
            f"Error: P1 detection did not complete (status: {p1_status}"
            f"{', ' + pass_info['error'] if pass_info.get('error') else ''}); "
            "refusing to run P2 over an incomplete artifact.",
            file=sys.stderr,
        )
        store.close()
        sys.exit(1)

    # ---- P2 Tracking Pass (always a full re-run, never checkpointed) ----
    print("--- P2 Tracking ---")
    orchestrator.run_pass_p2()

    p2_info = store._manifest.get("passes", {}).get(OfflineOrchestrator.P2_PASS_NAME, {})
    p2_stats = p2_info.get("stats", {})
    print(f"  Frames:         {p2_stats.get('frames_processed', '?')}/{meta.total_frames}")
    print(f"  Tracklet rows:  {p2_stats.get('total_tracklet_rows', '?')}")
    print(f"  Time:           {p2_stats.get('elapsed_s', '?')}s")
    print(f"  Effective fps:  {p2_stats.get('fps_effective', 0)}")

    if args.no_p3:
        store.close()
        total_elapsed = time.monotonic() - t0
        print()
        print(f"Total: {total_elapsed:.1f}s")
        print(f"Sidecar: {store.run_dir}")
        return

    # ---- P3 Identity Pass (global tracklet association) ----
    print()
    print("--- P3 Identity ---")
    result = orchestrator.run_pass_p3()

    if result is None:
        p3_info = store._manifest.get("passes", {}).get(OfflineOrchestrator.P3_PASS_NAME, {})
        print(
            f"Error: P3 did not run ({p3_info.get('error', 'unknown')}).",
            file=sys.stderr,
        )
        sys.exit(1)

    p3_info = store._manifest.get("passes", {}).get(OfflineOrchestrator.P3_PASS_NAME, {})
    p3_stats = p3_info.get("stats", {})
    print(f"  Tracklets in:   {p3_stats.get('tracklets_in', '?')}")
    print(f"  Persons:        {p3_stats.get('persons_out', '?')}")
    print(f"  Time:           {p3_stats.get('elapsed_s', '?')}s")
    print()
    # Honest unique-person count: confirmed persons + an uncertainty band from
    # below-threshold / gate-blocked near-merges. Same-clothing confusion is a
    # fundamental limit of any appearance method (DECISIONS B4) — the tool's
    # job is to surface the ambiguity, never hide it behind one precise number.
    confirmed = result.confirmed_count
    uncertain = result.uncertain_merges
    if uncertain:
        print(f"Unika personer:   {confirmed} unika, varav {uncertain} osäkra sammanslagningar")
    else:
        print(f"Unika personer:   {confirmed} unika")
    print(
        f"  (totalt {len(result.persons)} identiteter, varav "
        f"{len(result.persons) - confirmed} transienta < {config.p3_confirm_s:g}s)"
    )

    # ---- P5 Event Derivation Pass ----
    if args.no_p5:
        store.close()
        total_elapsed = time.monotonic() - t0
        print()
        print(f"Total: {total_elapsed:.1f}s")
        print(f"Sidecar: {store.run_dir}")
        return

    print()
    print("--- P5 Events ---")
    n_events = orchestrator.run_pass_p5()

    p5_info = store._manifest.get("passes", {}).get(OfflineOrchestrator.P5_PASS_NAME, {})
    p5_stats = p5_info.get("stats", {})
    by_cat = p5_stats.get("by_category", {})
    print(f"  Events:         {n_events}")
    if by_cat:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items()))
        print(f"  Per kategori:   {breakdown}")
    print(f"  Time:           {p5_stats.get('elapsed_s', '?')}s")

    store.close()
    total_elapsed = time.monotonic() - t0
    print()
    print(f"Total: {total_elapsed:.1f}s")
    print(f"Sidecar: {store.run_dir}")


if __name__ == "__main__":
    main()
