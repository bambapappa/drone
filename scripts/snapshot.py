"""Grab one annotated frame from a running server, without a browser.

Renders the same overlay as the web client (boxes, status colors, hazards,
base suggestion) onto the streamed JPEG — handy for debugging over SSH and
for verifying what clients will see.

Usage: python scripts/snapshot.py [--url http://localhost:8000] [--out snapshot.jpg]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct

import cv2
import numpy as np
import websockets

COLORS = {  # BGR
    "ok": (113, 204, 46),
    "still": (87, 71, 255),
    "toward_danger": (2, 165, 255),
    "threat": (56, 56, 255),
    "base": (255, 195, 52),
    "danger": (87, 71, 255),
    "smoke": (190, 180, 170),
    "fire": (53, 107, 255),
}
STATUS_TEXT = {"still": "STILLA", "toward_danger": "MOT FARA"}


def render(meta: dict, jpeg: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    lw = max(2, W // 480)

    for p in meta["persons"]:
        x, y, w, h = p["box"]
        color = COLORS.get(p["st"], COLORS["ok"])
        pt1, pt2 = (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H))
        cv2.rectangle(img, pt1, pt2, color, lw)
        label = f"P{p['pid']}"
        if p["st"] in STATUS_TEXT:
            label += f" {STATUS_TEXT[p['st']]}"
        cv2.putText(img, label, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        for a, b in zip(p["trail"], p["trail"][1:]):
            cv2.line(
                img,
                (int(a[0] * W), int(a[1] * H)),
                (int(b[0] * W), int(b[1] * H)),
                color,
                1,
            )

    for t in meta["threats"]:
        x, y, w, h = t["box"]
        cv2.rectangle(
            img,
            (int(x * W), int(y * H)),
            (int((x + w) * W), int((y + h) * H)),
            COLORS["threat"],
            lw * 2,
        )
        cv2.putText(
            img,
            f"! {t['cls'].upper()}",
            (int(x * W), int(y * H) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLORS["threat"],
            2,
        )

    hz = meta["hazards"]
    if hz["fire"]:
        c = (int(hz["fire"]["pos"][0] * W), int(hz["fire"]["pos"][1] * H))
        cv2.drawMarker(img, c, COLORS["fire"], cv2.MARKER_TRIANGLE_UP, 22, 3)
        cv2.putText(img, "BRAND", (c[0] + 14, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["fire"], 2)
    if hz["smoke"]:
        c = (int(hz["smoke"]["pos"][0] * W), int(hz["smoke"]["pos"][1] * H))
        dx, dy = hz["smoke"]["drift"]
        mag = (dx * dx + dy * dy) ** 0.5
        if mag > 1e-4:
            k = min(0.25, mag * 40) * W
            tip = (int(c[0] + dx / mag * k), int(c[1] + dy / mag * k))
            cv2.arrowedLine(img, c, tip, COLORS["smoke"], 2, tipLength=0.25)
        cv2.putText(img, "ROK", (c[0] + 6, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["smoke"], 2)

    if meta["base"]:
        c = (int(meta["base"]["pos"][0] * W), int(meta["base"]["pos"][1] * H))
        cv2.drawMarker(img, c, COLORS["base"], cv2.MARKER_DIAMOND, 22, 3)
        cv2.putText(
            img, "BAS (forslag)", (c[0] + 14, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["base"], 2
        )

    if meta["danger"]:
        c = (int(meta["danger"]["pos"][0] * W), int(meta["danger"]["pos"][1] * H))
        cv2.drawMarker(img, c, COLORS["danger"], cv2.MARKER_TILTED_CROSS, 26, 3)
        cv2.putText(img, "FARA", (c[0] + 16, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["danger"], 2)

    s = meta["stats"]
    hud = f"synliga {s['visible']}  unika {s['unique']}  irrationella {s['irr_now']}"
    if s["threat"]:
        hud += "  !! HOT !!"
    cv2.rectangle(img, (0, 0), (W, 28), (20, 16, 12), -1)
    cv2.putText(img, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 237, 232), 2)
    return img


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", default="snapshot.jpg")
    ap.add_argument("--wait", type=float, default=0.0, help="sekunder att vänta före bilden")
    args = ap.parse_args()

    ws_url = args.url.replace("http", "ws") + "/ws/stream"
    async with websockets.connect(ws_url, max_size=10_000_000) as ws:
        if args.wait:
            loop = asyncio.get_event_loop()
            end = loop.time() + args.wait
            while loop.time() < end:
                await ws.recv()
        pkt = await ws.recv()
    (n,) = struct.unpack(">I", pkt[:4])
    meta = json.loads(pkt[4 : 4 + n])
    img = render(meta, pkt[4 + n :])
    cv2.imwrite(args.out, img)
    print(f"Skrev {args.out} — {len(meta['persons'])} personer, stats: {meta['stats']}")


if __name__ == "__main__":
    asyncio.run(main())
