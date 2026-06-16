"""WebSocket stream + REST control API."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from app.core.config import settings
from app.vision.broadcast import broadcaster
from app.vision.pipeline import PipelineManager
from app.vision.sources import VIDEO_EXTS, is_stream_url, list_videos

router = APIRouter()
manager = PipelineManager(settings, broadcaster)

# Offline-analysis subprocesses, keyed by bundle name, so a second request
# doesn't start a duplicate run while one is in flight.
_analysis_procs: dict[str, subprocess.Popen] = {}


class SourceReq(BaseModel):
    # Either a file name inside video_dir, or a stream URL.
    name: str


class DangerReq(BaseModel):
    x: float
    y: float


class AnalyzeReq(BaseModel):
    name: str  # video file name inside video_dir
    stride: int = 1


def resolve_source(name: str) -> str:
    """Map a requested source to something safe to open."""
    if is_stream_url(name) or name.isdigit():
        return name
    p = (Path(settings.video_dir) / Path(name).name).resolve()
    base = Path(settings.video_dir).resolve()
    if not str(p).startswith(str(base)) or p.suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(400, "Ogiltig källa")
    if not p.is_file():
        raise HTTPException(404, f"Filen finns inte: {p.name}")
    return str(p)


def default_source() -> str | None:
    if settings.source:
        return settings.source
    vids = list_videos(settings.video_dir)
    if vids:
        return str(Path(settings.video_dir) / vids[0]["name"])
    return None


@router.get("/api/videos")
async def videos():
    return {"videos": list_videos(settings.video_dir)}


@router.get("/api/state")
async def state():
    return {
        "pipeline": manager.state(),
        "config": {
            "model": settings.model,
            "imgsz": settings.imgsz,
            "max_fps": settings.max_fps,
            "loop": settings.loop,
        },
    }


@router.post("/api/source")
async def set_source(req: SourceReq):
    src = resolve_source(req.name.strip())
    await asyncio.to_thread(manager.start, src)
    return {"ok": True, "source": src}


@router.post("/api/danger")
async def set_danger(req: DangerReq):
    p = manager.pipeline
    if p is None:
        raise HTTPException(409, "Ingen aktiv pipeline")
    p.set_danger_norm((min(max(req.x, 0.0), 1.0), min(max(req.y, 0.0), 1.0)))
    return {"ok": True}


@router.delete("/api/danger")
async def clear_danger():
    p = manager.pipeline
    if p is not None:
        p.set_danger_norm(None)
    return {"ok": True}


@router.post("/api/upload")
async def upload(file: UploadFile):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTS:
        raise HTTPException(400, f"Filtyp stöds inte: {suffix}")
    safe = Path(file.filename).name
    dest = Path(settings.video_dir) / safe
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > 2_000_000_000:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "Filen är för stor (max 2 GB)")
            f.write(chunk)
    return {"ok": True, "name": safe, "size_mb": round(size / 1e6, 1)}


# ---------- Offline analysis (after-action review) ----------


def _bundle_dir(name: str) -> Path:
    """Resolve a bundle directory safely under analyses_dir."""
    base = Path(settings.analyses_dir).resolve()
    p = (base / Path(name).name).resolve()
    if not str(p).startswith(str(base)):
        raise HTTPException(400, "Ogiltigt analysnamn")
    return p


def _read_state(d: Path) -> dict:
    f = d / "state.json"
    if f.is_file():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    # No state file but a finished bundle present.
    return {"status": "done" if (d / "meta.json").is_file() else "unknown"}


@router.get("/api/analyses")
async def analyses():
    base = Path(settings.analyses_dir)
    out = []
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            st = _read_state(d)
            meta = {}
            mf = d / "meta.json"
            if mf.is_file():
                try:
                    meta = json.loads(mf.read_text())
                except Exception:
                    meta = {}
            out.append(
                {
                    "name": d.name,
                    "status": st.get("status"),
                    "pct": st.get("pct"),
                    "source": meta.get("source"),
                    "frames_analyzed": meta.get("frames_analyzed"),
                    "created": meta.get("created"),
                }
            )
    return {"analyses": out}


@router.get("/api/analysis/{name}")
async def analysis_meta(name: str):
    f = _bundle_dir(name) / "meta.json"
    if not f.is_file():
        raise HTTPException(404, "Analysen finns inte")
    return json.loads(f.read_text())


@router.get("/api/analysis/{name}/frames")
async def analysis_frames(name: str):
    f = _bundle_dir(name) / "frames.jsonl"
    if not f.is_file():
        raise HTTPException(404, "Inga rutor")
    return PlainTextResponse(f.read_text(), media_type="application/x-ndjson")


@router.get("/api/analysis/{name}/events")
async def analysis_events(name: str):
    f = _bundle_dir(name) / "events.json"
    if not f.is_file():
        return {"events": []}
    return json.loads(f.read_text())


@router.get("/api/analysis/{name}/state")
async def analysis_state(name: str):
    return _read_state(_bundle_dir(name))


@router.get("/api/analysis/{name}/video")
async def analysis_video(name: str, request: Request):
    """Serve the bundle's source video (FileResponse handles Range requests,
    so the player can seek/scrub natively)."""
    d = _bundle_dir(name)
    mf = d / "meta.json"
    if not mf.is_file():
        raise HTTPException(404, "Analysen finns inte")
    src = json.loads(mf.read_text()).get("source", "")
    video = (Path(settings.video_dir) / Path(src).name).resolve()
    if not video.is_file():
        raise HTTPException(404, f"Källfilmen saknas: {src}")
    return FileResponse(str(video))


@router.post("/api/analyze")
async def analyze(req: AnalyzeReq):
    """Launch a non-real-time analysis of a video as a background process.
    Progress is polled via /api/analysis/{name}/state."""
    src = resolve_source(req.name.strip())  # validates + resolves to a real file
    name = Path(src).stem
    running = _analysis_procs.get(name)
    if running is not None and running.poll() is None:
        return {"ok": True, "name": name, "already_running": True}
    out = _bundle_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[2] / "scripts" / "analyze_offline.py"
    stride = max(1, min(10, req.stride))
    proc = subprocess.Popen(
        [sys.executable, str(script), src, "--out", str(out), "--stride", str(stride)],
        cwd=str(Path(settings.analyses_dir).resolve().parent),
    )
    _analysis_procs[name] = proc
    return {"ok": True, "name": name}


@router.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    broadcaster.attach(asyncio.get_running_loop())
    q = broadcaster.register()
    try:
        while True:
            data = await q.get()
            await ws.send_bytes(data)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        broadcaster.unregister(q)
