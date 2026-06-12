"""Generate a demo/test clip when no real footage is at hand.

Pans a virtual camera over a still photo containing real people (default:
the ultralytics sample image) and adds a synthetic fire + drifting smoke
patch so every analysis stage has something to chew on. The clip is a test
asset — the analysis pipeline itself makes no assumptions about it.

Usage:
    python scripts/make_demo_video.py [--image path] [--out videos/demo.mp4]
                                      [--seconds 24] [--fps 25]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def load_scene(image_path: str | None, canvas_w: int = 2200, canvas_h: int = 1080) -> np.ndarray:
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise SystemExit(f"Kan inte läsa bild: {image_path}")
    else:
        from ultralytics.utils import ASSETS

        img = cv2.imread(str(ASSETS / "bus.jpg"))

    scale = canvas_h / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * scale), canvas_h))

    rng = np.random.default_rng(7)
    canvas = np.full((canvas_h, canvas_w, 3), 84, dtype=np.uint8)
    noise = rng.integers(-25, 25, size=(canvas_h // 4, canvas_w // 4, 3))
    canvas = np.clip(
        canvas.astype(np.int16) + cv2.resize(noise.astype(np.float32), (canvas_w, canvas_h)),
        0,
        255,
    ).astype(np.uint8)

    x0 = (canvas_w - img.shape[1]) // 2
    canvas[:, x0 : x0 + img.shape[1]] = img
    return canvas


def add_fire_and_smoke(frame: np.ndarray, t: float, rng: np.random.Generator) -> None:
    h, w = frame.shape[:2]
    fx, fy = int(w * 0.82), int(h * 0.78)
    # flickering fire blob
    for _ in range(28):
        r = int(abs(rng.normal(14, 7))) + 4
        ox, oy = int(rng.normal(0, 22)), int(rng.normal(0, 14))
        color = (int(rng.uniform(0, 45)), int(rng.uniform(60, 130)), int(rng.uniform(200, 255)))
        cv2.circle(frame, (fx + ox, fy + oy), r, color, -1)
    # smoke drifting left-up from the fire
    drift = 90 * t
    for i in range(26):
        k = i / 26.0
        sx = int(fx - 30 - drift * (0.35 + 0.65 * k) - 60 * k + 18 * math.sin(t * 2 + i))
        sy = int(fy - 50 - 200 * k + 12 * math.cos(t * 1.6 + i * 0.7))
        r = int(26 + 46 * k)
        if sx < -r or sy < -r:
            continue
        g = int(135 + 45 * math.sin(t * 3 + i))
        overlay = frame.copy()
        cv2.circle(overlay, (sx, sy), r, (g, g, g), -1)
        a = 0.5 * (1 - k * 0.55)
        cv2.addWeighted(overlay, a, frame, 1 - a, 0, frame)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="scenbild (default: ultralytics bus.jpg)")
    ap.add_argument("--out", default="videos/demo.mp4")
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    scene = load_scene(args.image)
    sh, sw = scene.shape[:2]
    vw, vh = args.width, args.height

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (vw, vh))
    rng = np.random.default_rng(11)
    n = int(args.seconds * args.fps)
    max_x = sw - vw - 1
    max_y = sh - vh - 1

    for i in range(n):
        t = i / args.fps
        # slow sinusoidal drone pan with slight vertical bob
        cx = 0.5 + 0.42 * math.sin(2 * math.pi * t / 20.0)
        cy = 0.5 + 0.25 * math.sin(2 * math.pi * t / 13.0 + 1.2)
        x = int(cx * max_x)
        y = int(cy * max_y)
        frame = scene[y : y + vh, x : x + vw].copy()
        add_fire_and_smoke(frame, t, rng)
        writer.write(frame)

    writer.release()
    print(f"Skrev {args.out}: {n} bildrutor, {vw}x{vh} @ {args.fps} fps")


if __name__ == "__main__":
    main()
