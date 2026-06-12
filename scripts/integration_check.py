"""End-to-end smoke check against a running server.

Connects to the websocket stream, collects packets for a while and verifies
the PoC promises: people detected with stable IDs, smooth box motion, behavior
flags appearing, hazards/base present, danger API working.

Usage: python scripts/integration_check.py [ws://localhost:8000]
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from collections import defaultdict

import httpx
import websockets

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
WS = BASE.replace("http", "ws") + "/ws/stream"
SECONDS = 14.0


def parse(packet: bytes) -> tuple[dict, int]:
    (n,) = struct.unpack(">I", packet[:4])
    return json.loads(packet[4 : 4 + n]), len(packet) - 4 - n


async def main() -> int:
    metas: list[dict] = []
    jpeg_sizes: list[int] = []
    async with websockets.connect(WS, max_size=10_000_000) as ws:
        loop = asyncio.get_event_loop()
        end = loop.time() + SECONDS
        while loop.time() < end:
            try:
                pkt = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                print("FAIL: inga paket på 5 s")
                return 1
            meta, jlen = parse(pkt)
            metas.append(meta)
            jpeg_sizes.append(jlen)

    n = len(metas)
    fps = n / SECONDS
    det_fps = metas[-1]["det_fps"]
    with_persons = sum(1 for m in metas if m["persons"])
    pids = {p["pid"] for m in metas for p in m["persons"]}
    statuses = defaultdict(int)
    for m in metas:
        for p in m["persons"]:
            statuses[p["st"]] += 1

    # Box smoothness: max center jump per pid between *consecutive* packets.
    # (A person leaving the frame and re-entering later is not a jump.)
    last_center: dict[int, tuple[int, float, float]] = {}
    max_jump = 0.0
    for i, m in enumerate(metas):
        for p in m["persons"]:
            x, y, w, h = p["box"]
            c = (x + w / 2, y + h / 2)
            prev = last_center.get(p["pid"])
            if prev is not None and prev[0] == i - 1:
                max_jump = max(max_jump, abs(c[0] - prev[1]) + abs(c[1] - prev[2]))
            last_center[p["pid"]] = (i, c[0], c[1])

    fire_seen = sum(1 for m in metas if m["hazards"]["fire"])
    smoke_seen = sum(1 for m in metas if m["hazards"]["smoke"])
    base_seen = sum(1 for m in metas if m["base"])
    unique = metas[-1]["stats"]["unique"]

    kb = sum(jpeg_sizes) // max(n, 1) // 1024
    print(f"paket: {n} ({fps:.1f} fps), analys: {det_fps} Hz, jpeg: {kb} kB/bild")
    print(f"paket med personer: {with_persons}/{n}, unika pid: {sorted(pids)}, totalt unika: {unique}")
    print(f"statusfördelning: {dict(statuses)}")
    print(f"max boxhopp (norm): {max_jump:.4f}")
    print(f"hazards: fire {fire_seen}/{n}, smoke {smoke_seen}/{n}, basförslag {base_seen}/{n}")

    async with httpx.AsyncClient(base_url=BASE) as c:
        r = await c.post("/api/danger", json={"x": 0.8, "y": 0.75})
        assert r.status_code == 200, r.text
        await asyncio.sleep(0.5)
    async with websockets.connect(WS, max_size=10_000_000) as ws:
        meta, _ = parse(await ws.recv())
        danger_ok = meta["danger"] is not None
    async with httpx.AsyncClient(base_url=BASE) as c:
        await c.delete("/api/danger")
    print(f"faropunkt via API: {'ok' if danger_ok else 'SAKNAS'}")

    failures = []
    if fps < 8:
        failures.append(f"för låg fps: {fps:.1f}")
    if with_persons < n * 0.5:
        failures.append("personer saknas i för många paket")
    if not pids:
        failures.append("inga personer alls")
    if len(pids) > 12:
        failures.append(f"id-explosion: {len(pids)} pid för en statisk scen")
    if max_jump > 0.08:
        failures.append(f"boxar hoppar: {max_jump:.3f}")
    if statuses.get("still", 0) == 0:
        failures.append("ingen STILLA-flagga trots statiska personer (stabilisering trasig?)")
    if fire_seen == 0:
        failures.append("brand ej detekterad i demoklippet")
    if base_seen == 0:
        failures.append("inget basförslag")
    if not danger_ok:
        failures.append("faropunkt-API svarar inte i strömmen")

    if failures:
        print("\nFAIL:\n - " + "\n - ".join(failures))
        return 1
    print("\nOK: alla integrationskontroller passerade")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
