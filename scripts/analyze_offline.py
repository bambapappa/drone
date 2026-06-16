"""Run non-real-time analysis over a video and write a replayable bundle.

Unlike the live server this detects on *every* frame (best annotation, no
flow approximation) and emits a bundle the web player scrubs through. Model
and thresholds come from the same env/.env config as the server, so you can
A/B a model or resolution against real footage and step through the result
frame by frame to judge what was missed.

Usage:
    python scripts/analyze_offline.py videos/film.mp4
    python scripts/analyze_offline.py videos/film.mp4 --out analyses/film --stride 2
    MODEL=models/visdrone-yolov8s.pt IMGSZ=1280 CONF=0.2 \
        python scripts/analyze_offline.py videos/film.mp4

Then open the player:  http://localhost:8000/player  (or make serve first).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow `python scripts/analyze_offline.py` to import the app package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="sökväg till filmen (t.ex. videos/film.mp4)")
    ap.add_argument("--out", default="", help="utmapp för analysen (default: analyses/<filnamn>)")
    ap.add_argument("--stride", type=int, default=1, help="analysera var N:te ruta (1 = alla)")
    args = ap.parse_args()

    from app.core.config import settings
    from app.vision.offline import OfflineAnalyzer

    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f"Filen finns inte: {video}")
    out = Path(args.out) if args.out else Path("analyses") / video.stem

    print(f"Analyserar {video} → {out}")
    print(f"  modell={settings.model} imgsz={settings.imgsz} conf={settings.conf} tiles={settings.tiles}")

    last = [time.monotonic()]

    def progress(done: int, total: int) -> None:
        now = time.monotonic()
        if now - last[0] < 1.0 and done < total:
            return
        last[0] = now
        pct = f"{100 * done / total:.0f}%" if total else f"{done}"
        print(f"  {pct} ({done}/{total or '?'})", end="\r", flush=True)

    analyzer = OfflineAnalyzer(settings, str(video))
    meta = analyzer.run(out, stride=args.stride, progress=progress)
    s = meta["summary"]
    print(
        f"\nKlart: {meta['frames_analyzed']} rutor analyserade, "
        f"{s['unique']} unika personer, max {s['max_visible']} synliga samtidigt, "
        f"{s['n_events']} händelser ({meta['analysis_wall_s']} s)."
    )
    print(f"Öppna spelaren: /player?a={out.name}")


if __name__ == "__main__":
    main()
