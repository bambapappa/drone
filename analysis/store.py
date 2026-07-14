"""Artifact store v1: versioned sidecar store for analysis results.

Schema (per architecture report §3):
  frames/            per frame: frame_no (PK), pts_ms, stab_offset [dx,dy]
  detections/        P1 output only — raw predict boxes, never tracker-
                     adjusted: frame_no, det_id, xyxy_raw, conf, class,
                     embedding_ref (raw per-detection appearance vector)
  tracklets/         P2 output — one row per (track_id, frame): tracklet_id,
                     frame_no, det_id (references back to detections/),
                     cls, conf, xyxy (tracker/Kalman-adjusted box). Tracker
                     output lives here; detections/xyxy_raw is never
                     overwritten by it.
  manifest.json      video hash, config hash, model+weights versions, seed,
                     code version, pass log

Each table is a directory of JSONL files, one per pass. The manifest ties
everything together and enables bit-identical re-runs.

Other tables (persons/, trajectories/, events/, annotations/) are later
phases' concern but the schema is designed so they slot in without
restructuring.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SIDECAR_VERSION = 1


def tracker_lib_version() -> str:
    """Version of the tracking library (ultralytics) that P2's BOTSORT
    tracking pass runs against. Recorded in the manifest so a P2 re-run's
    determinism guarantee (byte-identical tracklets given the same P1
    output) is provenance-checked the same way P1's resume guard is."""
    try:
        import ultralytics

        return ultralytics.__version__
    except ImportError:
        return "no-ultralytics"


def code_version() -> str:
    """Git commit SHA of the running code, plus a '-dirty' suffix if the
    working tree has uncommitted changes. 'unknown' outside a git checkout.

    Used as part of the resume validation guard: resuming across a code
    change would silently build a chimera artifact, so a resumed run must
    match the code version that started it.
    """
    try:
        repo_dir = Path(__file__).resolve().parent
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo_dir,
        )
        if sha.returncode != 0:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo_dir,
        )
        suffix = "-dirty" if dirty.returncode == 0 and dirty.stdout.strip() else ""
        return sha.stdout.strip() + suffix
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class ResumeValidationError(ValueError):
    """Raised when --resume targets a run whose provenance doesn't match
    the current invocation (video, config, or code version changed)."""


class ArtifactStore:
    """Versioned sidecar store for a single analysis run.

    Directory layout:
      <output_dir>/
        <run_id>/           # uuid-based run directory
          manifest.json
          frames/           # frame-level data (JSONL per pass)
          detections/       # P1's raw per-detection data (JSONL per pass)
          tracklets/        # P2's per-(track_id, frame) tracked boxes (JSONL per pass)
          checkpoints/      # P1 checkpoint state (P2 is never checkpointed)
    """

    def __init__(self, output_dir: str, video_hash: str, config_hash: str, run_id: str | None = None):
        self.output_dir = Path(output_dir)
        self.video_hash = video_hash
        self.config_hash = config_hash
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.run_dir = self.output_dir / self.run_id
        self._manifest: dict[str, Any] = {}
        self._open_writers: dict[str, Any] = {}

    def create(self) -> None:
        """Create the sidecar directory structure and initial manifest.

        Always mints a fresh run — the default, provenance-clean path. Never
        looks up or reuses an existing run directory; that is exclusively
        open_existing()'s job, and only when explicitly requested via --resume.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("frames", "detections", "tracklets", "checkpoints"):
            (self.run_dir / sub).mkdir(exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()
        self._manifest = {
            "sidecar_version": SIDECAR_VERSION,
            "run_id": self.run_id,
            "created_at": now,
            "video_hash": self.video_hash,
            "config_hash": self.config_hash,
            "code_version": code_version(),
            "tracker_lib_version": tracker_lib_version(),
            "passes": {},
            "invocations": [{"timestamp": now, "code_version": code_version(), "resumed_from": None}],
        }
        self._write_manifest()

    @classmethod
    def open_existing(cls, output_dir: str, run_id: str, video_hash: str, config_hash: str) -> ArtifactStore:
        """Open an existing run directory for explicit --resume.

        Refuses to continue unless the target manifest's video_hash,
        config_hash, code_version, and tracker_lib_version all match the
        current invocation — resuming across a video, config (which includes
        model/weights), code, or tracker-library change would build a
        chimera artifact the manifest would then misrepresent as a single
        consistent run. Raises ResumeValidationError on any mismatch or if
        the run directory doesn't exist.
        """
        run_dir = Path(output_dir) / run_id
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise ResumeValidationError(f"No run found at {run_dir} (manifest.json missing)")

        with open(manifest_path) as f:
            manifest = json.load(f)

        current_code_version = code_version()
        current_tracker_lib_version = tracker_lib_version()
        mismatches = []
        if manifest.get("video_hash") != video_hash:
            mismatches.append(
                f"video_hash: run has {manifest.get('video_hash')!r}, current is {video_hash!r}"
            )
        if manifest.get("config_hash") != config_hash:
            mismatches.append(
                f"config_hash: run has {manifest.get('config_hash')!r}, current is {config_hash!r}"
            )
        if manifest.get("code_version") != current_code_version:
            mismatches.append(
                f"code_version: run has {manifest.get('code_version')!r}, current is {current_code_version!r}"
            )
        if manifest.get("tracker_lib_version") != current_tracker_lib_version:
            mismatches.append(
                f"tracker_lib_version: run has {manifest.get('tracker_lib_version')!r}, "
                f"current is {current_tracker_lib_version!r}"
            )
        if mismatches:
            raise ResumeValidationError(
                f"Cannot resume run {run_id}: provenance mismatch\n  " + "\n  ".join(mismatches)
            )

        store = cls(output_dir, video_hash, config_hash, run_id=run_id)
        store._manifest = manifest
        store._manifest.setdefault("invocations", [])
        store._manifest["invocations"].append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "code_version": current_code_version,
                "resumed_from": run_id,
            }
        )
        store._write_manifest()
        return store

    @staticmethod
    def resolve_latest(output_dir: str, video_hash: str, config_hash: str) -> str | None:
        """Find the most recently created run under output_dir whose
        video_hash and config_hash match, for the explicit `--resume latest`
        convenience. Returns None if no matching run exists.

        Resolution only happens when the user explicitly passes
        `--resume latest` — a bare `analyze <video>` never consults this.
        """
        base = Path(output_dir)
        if not base.is_dir():
            return None
        best_run_id: str | None = None
        best_created_at = ""
        for candidate in base.iterdir():
            manifest_path = candidate / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if manifest.get("video_hash") != video_hash or manifest.get("config_hash") != config_hash:
                continue
            created_at = manifest.get("created_at", "")
            if created_at > best_created_at:
                best_created_at = created_at
                best_run_id = manifest.get("run_id", candidate.name)
        return best_run_id

    def record_pass_start(self, pass_name: str, pass_meta: dict[str, Any]) -> None:
        """Record that a pass has started. Called before processing begins.

        Preserves the prior invocation's checkpoint watermark (last_frame), if
        any, so the manifest stays self-describing across a resumed run rather
        than losing progress history the instant the new invocation starts.
        """
        prior = self._manifest["passes"].get(pass_name, {})
        self._manifest["passes"][pass_name] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "meta": pass_meta,
            "status": "running",
            "last_frame": prior.get("last_frame", -1),
        }
        self._write_manifest()

    def record_pass_complete(self, pass_name: str, stats: dict[str, Any]) -> None:
        """Record that a pass has completed successfully."""
        if pass_name in self._manifest["passes"]:
            self._manifest["passes"][pass_name]["completed_at"] = datetime.now(timezone.utc).isoformat()
            self._manifest["passes"][pass_name]["status"] = "complete"
            self._manifest["passes"][pass_name]["stats"] = stats
        self._write_manifest()

    def record_pass_error(self, pass_name: str, error: str) -> None:
        """Record pass failure."""
        if pass_name in self._manifest["passes"]:
            self._manifest["passes"][pass_name]["status"] = "error"
            self._manifest["passes"][pass_name]["error"] = error
        self._write_manifest()

    def add_frame(self, pass_name: str, frame_no: int, data: dict[str, Any]) -> None:
        """Write a frame record to frames/<pass_name>.jsonl.

        The record is append-only JSONL — simple, inspectable, resumable.
        `data` must contain at minimum: {'frame_no': int, 'pts_ms': float}.
        """
        fpath = self.run_dir / "frames" / f"{pass_name}.jsonl"
        data["frame_no"] = frame_no
        json_line = json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n"
        with open(fpath, "a") as f:
            f.write(json_line)

    def add_detection(self, pass_name: str, frame_no: int, det_id: int, data: dict[str, Any]) -> None:
        """Write a detection record to detections/<pass_name>.jsonl.

        `data` must contain at minimum:
        {'frame_no': int, 'det_id': int, 'xyxy_raw': [x0,y0,x1,y1], 'conf': float,
         'cls': str, 'embedding': [...]}
        The embedding is the raw per-detection appearance vector (HSV histogram
        in Phase 0 — this is the cache the architecture report identifies as the
        hard blocker for global re-ID in later phases).
        """
        fpath = self.run_dir / "detections" / f"{pass_name}.jsonl"
        data["frame_no"] = frame_no
        data["det_id"] = det_id
        json_line = json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n"
        with open(fpath, "a") as f:
            f.write(json_line)

    def start_fresh_pass_output(self, table: str, pass_name: str) -> None:
        """Truncate a per-pass JSONL table before a pass that always fully
        re-runs (e.g. P2, which is never resumed — it's cheap and
        deterministic given the same P1 output). Without this, re-running
        the pass would append to stale output from a prior invocation."""
        fpath = self.run_dir / table / f"{pass_name}.jsonl"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text("")

    def add_tracklet_frame(
        self, pass_name: str, tracklet_id: int, frame_no: int, det_id: int, data: dict[str, Any]
    ) -> None:
        """Write one (tracklet, frame) row to tracklets/<pass_name>.jsonl.

        `data` must contain at minimum: {'cls': str, 'conf': float,
        'xyxy': [x0,y0,x1,y1]} — the tracker/Kalman-adjusted box. This is
        the only place tracker output is persisted; detections/xyxy_raw is
        P1's pure predict output and is never overwritten by it.
        """
        fpath = self.run_dir / "tracklets" / f"{pass_name}.jsonl"
        data = dict(data)
        data["tracklet_id"] = tracklet_id
        data["frame_no"] = frame_no
        data["det_id"] = det_id
        json_line = json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n"
        with open(fpath, "a") as f:
            f.write(json_line)

    def iter_tracklets(self, pass_name: str):
        """Yield persisted tracklet-frame rows for a pass, in write order."""
        fpath = self.run_dir / "tracklets" / f"{pass_name}.jsonl"
        if not fpath.exists():
            return
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def save_checkpoint(self, pass_name: str, state: dict[str, Any]) -> str:
        """Save a checkpoint for the given pass. Returns the checkpoint path."""
        cp_dir = self.run_dir / "checkpoints" / pass_name
        cp_dir.mkdir(parents=True, exist_ok=True)
        cp_path = cp_dir / "checkpoint.json"
        with open(cp_path, "w") as f:
            json.dump(state, f, indent=2, default=_json_default)
        return str(cp_path)

    def load_checkpoint(self, pass_name: str) -> dict[str, Any] | None:
        """Load the latest checkpoint for the given pass, or None."""
        cp_path = self.run_dir / "checkpoints" / pass_name / "checkpoint.json"
        if not cp_path.exists():
            return None
        with open(cp_path) as f:
            return json.load(f)

    def get_last_frame(self, pass_name: str) -> int:
        """Return the last durably-processed frame_no for the given pass, or -1.

        Derived from frames/ alone: add_frame() is called exactly once per
        processed frame, after all of that frame's detections have already
        been written, so a frame row is only ever present once the frame is
        fully durable. Including detections/ in this scan would misreport a
        frame as complete if a crash left a partial or orphaned run of
        detection rows with no matching frame row.
        """
        last_frame = -1
        fpath = self.run_dir / "frames" / f"{pass_name}.jsonl"
        if not fpath.exists():
            return last_frame
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    last_frame = max(last_frame, json.loads(line).get("frame_no", -1))
        return last_frame

    def discard_orphaned_detections(self, pass_name: str, last_frame: int) -> int:
        """Drop any detection rows for frame_no > last_frame.

        Detection rows are appended before the owning frame's row, so a crash
        mid-frame can leave orphaned detection rows for a frame that never got
        a frame row (and therefore isn't counted as processed by
        get_last_frame). Call this before recomputing det_id on resume so
        those orphaned rows aren't double-counted or left with an ID that
        collides with the frame's about-to-be-redone detections.

        Returns the number of rows discarded.
        """
        fpath = self.run_dir / "detections" / f"{pass_name}.jsonl"
        if not fpath.exists():
            return 0
        kept: list[str] = []
        discarded = 0
        with open(fpath) as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if json.loads(stripped).get("frame_no", -1) > last_frame:
                    discarded += 1
                else:
                    kept.append(line if line.endswith("\n") else line + "\n")
        if discarded:
            with open(fpath, "w") as f:
                f.writelines(kept)
        return discarded

    def record_pass_progress(self, pass_name: str, last_frame: int) -> None:
        """Update the manifest's per-pass checkpoint watermark.

        Purely descriptive (resume itself is derived from frames/ on disk,
        not this field) — makes a partially-completed run self-describing by
        inspecting manifest.json alone, without needing to replay the JSONL.
        """
        if pass_name not in self._manifest["passes"]:
            return
        entry = self._manifest["passes"][pass_name]
        entry["last_frame"] = last_frame
        if entry.get("status") == "running":
            entry["status"] = "partial"
        self._write_manifest()

    def get_max_det_id(self, pass_name: str) -> int:
        """Return the highest persisted det_id for the given pass, or -1.

        Read directly from detections/<pass>.jsonl rather than a checkpoint:
        checkpoints are only written periodically, but detection records are
        appended immediately per-frame, so this reflects every det_id actually
        on disk even if the crash happened between two checkpoints.
        """
        max_det_id = -1
        fpath = self.run_dir / "detections" / f"{pass_name}.jsonl"
        if not fpath.exists():
            return max_det_id
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    max_det_id = max(max_det_id, json.loads(line).get("det_id", -1))
        return max_det_id

    def iter_detections(self, pass_name: str, max_frame_no: int | None = None):
        """Yield persisted P1 detection records for a pass, in write order
        (which is det_id order), optionally bounded to frame_no <=
        max_frame_no. P2 streams these — grouped by frame_no — to drive the
        tracking pass without re-running inference.
        """
        fpath = self.run_dir / "detections" / f"{pass_name}.jsonl"
        if not fpath.exists():
            return
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if max_frame_no is not None and rec.get("frame_no", -1) > max_frame_no:
                    continue
                yield rec

    def close(self) -> None:
        self._write_manifest()

    def _write_manifest(self) -> None:
        with open(self.run_dir / "manifest.json", "w") as f:
            json.dump(self._manifest, f, indent=2, default=str)

    @staticmethod
    def config_hash_from_settings(settings: dict[str, Any]) -> str:
        """Deterministic hash of analysis settings for provenance."""
        canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _json_default(obj: Any) -> Any:
    """Handle numpy types in JSON serialization."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
