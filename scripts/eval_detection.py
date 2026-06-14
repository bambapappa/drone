"""Compare detection models/settings on sampled frames from a video.

Measures what matters for the PoC on real footage: how many people each
model finds, at what confidence, and how fast — so defaults are chosen
from evidence, not guesses.

Usage:
    python scripts/eval_detection.py videos/film.mp4 \
        --models yolo11n.pt models/visdrone-yolov8s.pt \
        --imgsz 640 960 --samples 30 [--conf 0.3] [--ignore 0.66,0,0.34,0.44]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# Allow `python scripts/eval_detection.py` to import the app package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HUMAN = {"person", "pedestrian", "people"}


def sample_frames(path: str, n: int) -> list:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


def in_ignore(cx, cy, w, h, regions) -> bool:
    nx, ny = cx / w, cy / h
    return any(rx <= nx <= rx + rw and ry <= ny <= ry + rh for rx, ry, rw, rh in regions)


def predict_tiled(model, frame, n, imgsz, conf, hids):
    """Per-tile prediction merged with global NMS; returns (xyxy, conf) lists.

    Uses the same tiling helpers as the production detector so the eval
    measures what the pipeline actually does.
    """
    from app.vision.tiling import nms_merge, tile_grid

    h, w = frame.shape[:2]
    boxes, scores = [], []
    for x0, y0, x1, y1 in tile_grid(w, h, n):
        res = model.predict(frame[y0:y1, x0:x1], imgsz=imgsz, conf=conf, classes=hids, verbose=False)[0]
        for b in res.boxes or []:
            bx0, by0, bx1, by1 = b.xyxy[0].tolist()
            boxes.append([bx0 + x0, by0 + y0, bx1 + x0, by1 + y0])
            scores.append(float(b.conf[0]))
    keep = nms_merge(boxes, scores, [0] * len(boxes))
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--models", nargs="+", default=["yolo11n.pt"])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640])
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--ignore", default="", help="';'-separerade x,y,w,h att exkludera")
    ap.add_argument("--tiles", type=int, default=1, help="NxN tiled inferens (1 = av)")
    args = ap.parse_args()

    from ultralytics import YOLO

    regions = [tuple(float(v) for v in p.split(",")) for p in args.ignore.split(";") if p.strip()]
    frames = sample_frames(args.video, args.samples)
    if not frames:
        raise SystemExit("Inga bildrutor kunde läsas")
    h, w = frames[0].shape[:2]
    print(f"{args.video}: {len(frames)} provrutor à {w}x{h}, conf {args.conf}")
    print(f"{'modell':<24}{'imgsz':>6}{'medel':>7}{'max':>5}{'rutor>0':>9}{'medelconf':>11}{'ms/ruta':>9}")

    for mpath in args.models:
        model = YOLO(mpath)
        lower = {i: n.lower() for i, n in model.names.items()}
        hids = [i for i, n in lower.items() if n in HUMAN]
        for imgsz in args.imgsz:
            model.predict(frames[0], imgsz=imgsz, verbose=False)  # warmup
            counts, confs, t0 = [], [], time.perf_counter()
            for fr in frames:
                if args.tiles > 1:
                    bxs, scs = predict_tiled(model, fr, args.tiles, imgsz, args.conf, hids)
                else:
                    res = model.predict(fr, imgsz=imgsz, conf=args.conf, classes=hids, verbose=False)[0]
                    bxs = [b.xyxy[0].tolist() for b in res.boxes or []]
                    scs = [float(b.conf[0]) for b in res.boxes or []]
                n_pers = 0
                for (x0, y0, x1, y1), sc in zip(bxs, scs):
                    if not in_ignore((x0 + x1) / 2, (y0 + y1) / 2, w, h, regions):
                        n_pers += 1
                        confs.append(sc)
                counts.append(n_pers)
            ms = (time.perf_counter() - t0) * 1000 / len(frames)
            mean = sum(counts) / len(counts)
            with_p = sum(1 for c in counts if c > 0)
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            name = mpath.split("/")[-1]
            print(
                f"{name:<24}{imgsz:>6}{mean:>7.1f}{max(counts):>5}"
                f"{with_p:>5}/{len(counts):<3}{avg_conf:>11.2f}{ms:>9.0f}"
            )


if __name__ == "__main__":
    main()
