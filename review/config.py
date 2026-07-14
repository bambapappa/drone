"""Settings for the review API.

Resolves the two on-disk locations the review layer reads/writes:
  - OUTPUT_DIR:  where the analysis CLI wrote sidecars (./analysis-output)
  - VIDEO_DIR:   where the original video files live (./videos)

Both are overridable via env vars so docker-compose can mount them at any
path without code changes, mirroring how the realtime PoC's VIDEO_DIR works.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReviewSettings:
    output_dir: Path
    video_dir: Path

    @classmethod
    def from_env(cls) -> "ReviewSettings":
        # Defaults match docker-compose.offline.yml's volume mounts so a bare
        # `uvicorn review.main:app` in the project root works without config.
        output_dir = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", "analysis-output")).resolve()
        video_dir = Path(os.environ.get("VIDEO_DIR", "videos")).resolve()
        return cls(output_dir=output_dir, video_dir=video_dir)


def get_settings() -> ReviewSettings:
    """Factory used as a FastAPI dependency. Settings are computed once per
    request, not cached — env-var lookups are cheap, and avoiding a global
    keeps the review API trivially test-configurable per test."""
    return ReviewSettings.from_env()
