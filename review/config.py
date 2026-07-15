"""Settings for the review API.

Resolves the two on-disk locations the review layer reads/writes:
  - OUTPUT_DIR:  where the analysis CLI wrote sidecars (./analysis-output)
  - VIDEO_DIR:   where the original video files live (./videos)

Both are overridable via env vars so docker-compose can mount them at any
path without code changes, mirroring how the realtime PoC's VIDEO_DIR works.

Phase 5 adds five independent feature toggles (report §5.5-5.9), each its own
env var so the captain can enable exactly the set that proves useful without
a code change (docker-compose passes them straight through):

  FEATURE_DOSSIER       persondossié + manual split/merge corrections
  FEATURE_GROUND_TRUTH  facit (reference truth) entry + AI/operator scoring
  FEATURE_RUN_COMPARE   diff two runs of the same video (multi-config)
  FEATURE_CLIP_EXPORT   annotated clip export around an event (client-side)
  FEATURE_HEATMAP       dwell/coverage heatmap layer

All default ON: every Phase 5 feature is a read-only or annotation-layer
capability with a harmless empty state, so shipping them visible makes them
discoverable; the toggles exist to switch OFF what turns out not to earn its
screen space. Each toggle is checked independently — disabling one never
affects another. One deliberate asymmetry: FEATURE_DOSSIER gates the dossier
UI and *new* correction writes, but identity corrections already recorded in
the annotation log keep applying to read-time projections even when the
toggle is off — recorded human review work is never silently hidden by a
config flag (same principle as re-analysis never touching annotations/).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Env values accepted as "on"/"off" for feature toggles. Anything else falls
# back to the default so a typo can't silently disable a feature.
_TRUTHY = {"1", "true", "on", "yes"}
_FALSY = {"0", "false", "off", "no"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    return default


@dataclass(frozen=True)
class ReviewSettings:
    output_dir: Path
    video_dir: Path
    # Phase 5 feature toggles (see module docstring for defaults rationale).
    feature_dossier: bool = True
    feature_ground_truth: bool = True
    feature_run_compare: bool = True
    feature_clip_export: bool = True
    feature_heatmap: bool = True

    @classmethod
    def from_env(cls) -> "ReviewSettings":
        # Defaults match docker-compose.offline.yml's volume mounts so a bare
        # `uvicorn review.main:app` in the project root works without config.
        output_dir = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", "analysis-output")).resolve()
        video_dir = Path(os.environ.get("VIDEO_DIR", "videos")).resolve()
        return cls(
            output_dir=output_dir,
            video_dir=video_dir,
            feature_dossier=_env_flag("FEATURE_DOSSIER", True),
            feature_ground_truth=_env_flag("FEATURE_GROUND_TRUTH", True),
            feature_run_compare=_env_flag("FEATURE_RUN_COMPARE", True),
            feature_clip_export=_env_flag("FEATURE_CLIP_EXPORT", True),
            feature_heatmap=_env_flag("FEATURE_HEATMAP", True),
        )

    def features_payload(self) -> dict[str, bool]:
        """Toggle state for the UI (`GET /api/features`) — the client hides
        the corresponding tabs/buttons for disabled features."""
        return {
            "dossier": self.feature_dossier,
            "ground_truth": self.feature_ground_truth,
            "run_compare": self.feature_run_compare,
            "clip_export": self.feature_clip_export,
            "heatmap": self.feature_heatmap,
        }


def get_settings() -> ReviewSettings:
    """Factory used as a FastAPI dependency. Settings are computed once per
    request, not cached — env-var lookups are cheap, and avoiding a global
    keeps the review API trivially test-configurable per test."""
    return ReviewSettings.from_env()
