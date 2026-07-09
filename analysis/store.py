"""Artifact store v1: versioned sidecar store for analysis results.

Schema (per architecture report §3):
  frames/            per frame: frame_no (PK), pts_ms, stab_offset [dx,dy]
  detections/        per detection: frame_no, det_id, xyxy_raw, conf, class,
                     embedding_ref (raw per-detection appearance vector)
  manifest.json      video hash, config hash, model+weights versions, seed,
                     code version, pass log

Each table is a directory of JSONL files, one per pass. The manifest ties
everything together and enables bit-identical re-runs.

For Phase 0, we persist frames/ and detections/ at minimum. Other tables
(frames/, tracklets/, persons/, trajectories/, events/, annotations/) are
later phases' concern but the schema is designed so they slot in without
restructuring.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SIDECAR_VERSION = 1


class ArtifactStore:
    """Versioned sidecar store for a single analysis run.

    Directory layout:
      <output_dir>/
        <run_id>/           # uuid-based run directory
          manifest.json
          frames/           # frame-level data (JSONL per pass)
          detections/       # per-detection data (JSONL per pass)
          checkpoints/      # pass checkpoint state
    """

    def __init__(self, output_dir: str, video_hash: str, config_hash: str):
        self.output_dir = Path(output_dir)
        self.video_hash = video_hash
        self.config_hash = config_hash
        self.run_id = uuid.uuid4().hex[:12]
        self.run_dir = self.output_dir / self.run_id
        self._manifest: dict[str, Any] = {}
        self._open_writers: dict[str, Any] = {}

    def create(self) -> None:
        """Create the sidecar directory structure and initial manifest."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("frames", "detections", "checkpoints"):
            (self.run_dir / sub).mkdir(exist_ok=True)

        self._manifest = {
            "sidecar_version": SIDECAR_VERSION,
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "video_hash": self.video_hash,
            "config_hash": self.config_hash,
            "passes": {},
        }
        self._write_manifest()

    def record_pass_start(self, pass_name: str, pass_meta: dict[str, Any]) -> None:
        """Record that a pass has started. Called before processing begins."""
        self._manifest["passes"][pass_name] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "meta": pass_meta,
            "status": "running",
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
        """Return the last persisted frame_no for the given pass, or -1.

        Checks both frames/ and detections/: detections/ only gets a row when
        a frame produced at least one detection, so a trailing run of
        empty-detection frames would otherwise look unprocessed on resume.
        """
        last_frame = -1
        for sub in ("frames", "detections"):
            fpath = self.run_dir / sub / f"{pass_name}.jsonl"
            if not fpath.exists():
                continue
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_frame = max(last_frame, json.loads(line).get("frame_no", -1))
        return last_frame

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
