#!/usr/bin/env python3
"""Offline drone video analysis CLI.

Usage:
    analyze <video> [--output DIR] [--model PATH] [--device DEVICE]
                    [--imgsz N] [--conf F] [--tiles N] [--seed N]
                    [--resume RUN_ID | --resume latest]

Runs ingest + P1 detection end-to-end. The analysis sidecar is written to
<output>/<run_id>/ with manifest.json, frames/, detections/, and checkpoints/.

By default every invocation mints a fresh run_id and a fresh sidecar —
re-running the same command does NOT resume automatically. To continue an
interrupted run, pass --resume <run_id> (or --resume latest to resolve the
most recent matching run under --output). Resume refuses to continue unless
the target run's video hash, config hash, and code version all match the
current invocation, so it can never silently splice together a run across a
config or code change.
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
    ap.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
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
    print(f"Conf/IoU:         {args.conf}/{args.iou}")
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
        conf=args.conf,
        iou=args.iou,
        tiles=args.tiles,
        seed=args.seed,
        track_buffer_s=args.track_buffer_s,
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
    print(f"Sidecar store:    {store.run_dir}")
    print(f"  Run ID:         {store.run_id}")
    print()

    # ---- P1 Detection Pass ----
    print("--- P1 Detection ---")
    from analysis.orchestrator import OfflineOrchestrator

    orchestrator = OfflineOrchestrator(meta, store, config)
    orchestrator.run_pass_p1()

    pass_info = store._manifest.get("passes", {}).get("p1_detect", {})
    stats = pass_info.get("stats", {})
    print(f"  Frames:         {stats.get('frames_processed', '?')}/{meta.total_frames}")
    print(f"  Detections:     {stats.get('total_detections', '?')}")
    print(f"  Time:           {stats.get('elapsed_s', '?')}s")
    eff = stats.get("fps_effective", 0)
    print(f"  Effective fps:  {eff}")

    store.close()
    total_elapsed = time.monotonic() - t0
    print()
    print(f"Total: {total_elapsed:.1f}s")
    print(f"Sidecar: {store.run_dir}")


if __name__ == "__main__":
    main()
