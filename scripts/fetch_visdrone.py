"""Fetch community VisDrone-trained YOLO weights into models/.

Run this on a machine with open internet (the weights live on Hugging Face):

    python scripts/fetch_visdrone.py            # yolov8s (good accuracy/speed)
    python scripts/fetch_visdrone.py --size n   # fastest, for weak CPUs
    python scripts/fetch_visdrone.py --url https://...  # any direct .pt URL

Then point the app at the file (already the right class names in .env defaults):

    MODEL=models/visdrone-yolov8s.pt
    HUMAN_CLASSES=pedestrian,people

Provenance note: these are third-party community weights (mshamrai/yolov8*-visdrone
on Hugging Face, Ultralytics AGPL ecosystem). A .pt file is a pickle — only load
weights you trust. Verify with --verify (needs ultralytics installed), which loads
the model and prints its class names without running inference.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

HF_REPOS = {
    "n": "mshamrai/yolov8n-visdrone",
    "s": "mshamrai/yolov8s-visdrone",
    "m": "mshamrai/yolov8m-visdrone",
    "l": "mshamrai/yolov8l-visdrone",
}
CANDIDATE_FILES = ["best.pt", "model.pt", "weights/best.pt"]
PINNED_S_URL = (
    "https://huggingface.co/mshamrai/yolov8s-visdrone/resolve/"
    "523ab5140acfe3fd7b0f1ab5084ebd942159fd5f/best.pt"
)
PINNED_S_SHA256 = "e3776314790b381f2eb08ed87ccec71e19a7a0308ac064adb704643b73d06947"


def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "drone-poc-fetch"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 200:
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            total = 0
            with dest.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    total += len(chunk)
                    print(f"\r{dest.name}: {total / 1e6:.0f} MB", end="", flush=True)
            print()
            return total > 1_000_000  # a real model is several MB
    except urllib.error.URLError as e:
        print(f"  ({url}: {e.reason})")
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(HF_REPOS), default="s")
    ap.add_argument("--url", default=None, help="direct .pt URL (overrides --size)")
    ap.add_argument("--out", default=None, help="output path (default models/visdrone-yolov8<size>.pt)")
    ap.add_argument("--verify", action="store_true", help="load the model and print class names")
    args = ap.parse_args()

    dest = Path(args.out) if args.out else Path("models") / f"visdrone-yolov8{args.size}.pt"
    expected_sha256 = PINNED_S_SHA256 if args.size == "s" and args.url is None else None
    if args.url:
        urls = [args.url]
    elif args.size == "s":
        urls = [PINNED_S_URL]
    else:
        urls = [f"https://huggingface.co/{HF_REPOS[args.size]}/resolve/main/{f}" for f in CANDIDATE_FILES]
    if dest.exists():
        if expected_sha256 and sha256(dest) != expected_sha256:
            print(f"Fel: SHA-256 för {dest} stämmer inte med den pinnade VisDrone-s-vikten.")
            return 1
        print(f"{dest} finns redan — verifierad")
    else:
        ok = False
        temp_dest = dest.with_name(f".{dest.name}.download")
        for url in urls:
            print(f"Försöker {url}")
            temp_dest.unlink(missing_ok=True)
            if download(url, temp_dest):
                if expected_sha256 and sha256(temp_dest) != expected_sha256:
                    print(f"Fel: SHA-256 för den hämtade vikten stämmer inte: {url}")
                    temp_dest.unlink(missing_ok=True)
                    return 1
                temp_dest.replace(dest)
                ok = True
                break
            temp_dest.unlink(missing_ok=True)
        if not ok:
            print(
                "\nKunde inte hämta vikterna automatiskt. Ladda ned .pt-filen manuellt\n"
                f"(t.ex. från https://huggingface.co/{HF_REPOS[args.size]}) och lägg den som {dest},\n"
                "eller kör om med --url <direktlänk>."
            )
            return 1

    print(f"\nKlart: {dest}")
    print("Sätt i .env:")
    print(f"  MODEL={dest}")
    print("  HUMAN_CLASSES=pedestrian,people")

    if args.verify:
        from ultralytics import YOLO

        names = YOLO(str(dest)).names
        print(f"Modellens klasser: {sorted(names.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
