"""Per-detection appearance embedding — Phase 1's identity substrate.

P1 computes one appearance vector per detection crop and persists it (the
cache the live registry.py only kept EMA-blended, which is the hard blocker
for global re-ID). P3 then clusters tracklets over these vectors.

Two methods, deliberately tagged per-detection so a consumer knows which was
used and can reason about its reliability:

- **osnet** — a dedicated ReID embedding (OSNet-class), the PoC-2 upgrade
  DECISIONS B4 anticipated. Loaded lazily from a configurable weights path
  (mirrors how the YOLO `model` is handled: a path on disk, recorded in the
  manifest, never auto-fetched with a guessed URL). Run in eval + no_grad so
  it is deterministic (CPU) and consumes no RNG.

- **hsv** — the carried-forward HSV torso histogram (registry.appearance_hist).
  This is the **deliberate fallback** for crops too small for any ReID
  model's input floor: 10 px people at altitude are below OSNet's 256×128
  input by an order of magnitude. Down-scaling a 10 px crop to 256 px does
  not manufacture identity information; the HSV histogram at least captures
  dominant clothing color. This is a documented degradation, not a bug — the
  `embedding_method` field states which detections used which method, and P3
  only compares embeddings within the same method space (an osnet vector and
  an hsv vector are different-dimensional and not comparable).

Tests never instantiate OSNetEmbedder (no weights in CI); they drive the
orchestrator through a FakeEmbedder, exactly as P1's detection path uses
FakeDetector. The real OSNet path is exercised only in a real run with
weights present — the same contract Phase 0's real YOLO Detector has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from analysis.registry import appearance_hist

# OSNet's standard ReID input. Crops are letterboxed/resized to this before
# the forward pass. Anything smaller than REID_FLOOR on its shortest side
# never reaches the model — it goes straight to the HSV fallback.
OSNET_INPUT_HW = (256, 128)

# A detection box in raw frame pixels.
Box = tuple[float, float, float, float]


@dataclass
class EmbeddingResult:
    vector: np.ndarray  # L2-normalized
    method: str  # "osnet" | "hsv"


class Embedder(Protocol):
    """Per-detection appearance embedder.

    Returns a normalized vector + its method, or None when the crop is too
    small/degenerate to embed even with the HSV fallback (appearance_hist
    returns None below ~4×8 px). None is persisted as a null embedding and
    that detection contributes nothing to P3 clustering.
    """

    def embed(self, frame_bgr: np.ndarray, box_xyxy: Box) -> EmbeddingResult | None: ...


def _crop_min_side(box_xyxy: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = box_xyxy
    return min(x1 - x0, y1 - y0)


def below_reid_floor(box_xyxy: tuple[float, float, float, float], floor: int) -> bool:
    """Pure gate: does this crop fall below the ReID model's input floor?

    Factored out (rather than inlined in OSNetEmbedder) so it is unit-testable
    without torch or model weights — the floor/fallback decision is the part
    of the embedding path that genuinely needs a regression guard, since a
    wrong floor silently routes everything to the weak HSV space.
    """
    return _crop_min_side(box_xyxy) < floor


def _hsv_embed(frame_bgr: np.ndarray, box_xyxy: tuple[float, float, float, float]) -> EmbeddingResult | None:
    hist = appearance_hist(frame_bgr, box_xyxy)
    if hist is None:
        return None
    return EmbeddingResult(vector=hist, method="hsv")


class HSVEmbedder:
    """Always-HSV embedder. Used when no ReID weights are configured/present,
    so the pipeline still produces a (weak but functional) appearance cache
    and P3 still clusters. The manifest records that only HSV was used."""

    def embed(self, frame_bgr: np.ndarray, box_xyxy: Box) -> EmbeddingResult | None:
        return _hsv_embed(frame_bgr, box_xyxy)


class OSNetEmbedder:
    """OSNet-primary with HSV fallback below the crop floor.

    The weights are a TorchScript model loaded via torch.jit.load —
    architecture-agnostic and version-stable, so any exported OSNet-class
    ReID model works without this module hard-coding a specific network
    definition that could drift from the checkpoint. Standard ReID
    preprocessing (resize to 256×128, ImageNet normalize), eval + no_grad
    forward, L2-normalized output.
    """

    def __init__(self, weights_path: str, floor: int = 32, device: str = "cpu"):
        self.weights_path = weights_path
        self.floor = floor
        self.device = device
        self._model: Any = None  # lazy

    def _ensure_loaded(self) -> Any:
        if self._model is None:
            import torch  # lazy: tests and the web app don't require it

            self._model = torch.jit.load(self.weights_path, map_location=self.device).eval()
        return self._model

    def embed(self, frame_bgr: np.ndarray, box_xyxy: Box) -> EmbeddingResult | None:
        # Deliberate, documented degradation: crops below the model's input
        # floor carry no identity information at the model's native resolution
        # — route them to the HSV fallback instead of fabricating an embedding
        # from an over-upscaled 10 px crop.
        if below_reid_floor(box_xyxy, self.floor):
            return _hsv_embed(frame_bgr, box_xyxy)
        return self._osnet_embed(frame_bgr, box_xyxy)

    def _osnet_embed(self, frame_bgr: np.ndarray, box_xyxy: Box) -> EmbeddingResult | None:
        import cv2  # declared dep (opencv-python-headless)
        import torch  # lazy

        model = self._ensure_loaded()
        x0, y0, x1, y1 = (int(v) for v in box_xyxy)
        H, W = frame_bgr.shape[:2]
        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, y0), min(H, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return _hsv_embed(frame_bgr, box_xyxy)
        crop = np.ascontiguousarray(frame_bgr[y0:y1, x0:x1])
        # Standard ReID preprocessing: resize to OSNet's 256×128 input, convert
        # BGR→RGB, scale to [0,1], ImageNet-normalize. Done with cv2/numpy rather
        # than torchvision so the only declared deps (opencv, numpy, torch) apply.
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(
            crop_rgb, (OSNET_INPUT_HW[1], OSNET_INPUT_HW[0]), interpolation=cv2.INTER_LINEAR
        )
        tensor = torch.from_numpy(crop_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = model(tensor).cpu().numpy().flatten().astype(np.float64)
        n = np.linalg.norm(feat)
        if n <= 0:
            return None
        return EmbeddingResult(vector=feat / n, method="osnet")


def make_embedder(config: Any) -> Embedder:
    """Pick the embedder for a run from the config.

    OSNet when reid_weights is configured AND present on disk (so a missing
    weights file degrades to HSV-only with a clear provenance note, rather
    than crashing a run that the manifest would then misrepresent as having
    used OSNet). Otherwise HSV-only.
    """
    from pathlib import Path

    weights = getattr(config, "reid_weights", None)
    if weights and Path(weights).is_file():
        return OSNetEmbedder(weights, floor=getattr(config, "reid_floor", 32), device=config.device)
    return HSVEmbedder()
