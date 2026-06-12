"""WebSocket stream + REST control API."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.core.config import settings
from app.vision.broadcast import broadcaster
from app.vision.pipeline import PipelineManager
from app.vision.sources import VIDEO_EXTS, is_stream_url, list_videos

router = APIRouter()
manager = PipelineManager(settings, broadcaster)


class SourceReq(BaseModel):
    # Either a file name inside video_dir, or a stream URL.
    name: str


class DangerReq(BaseModel):
    x: float
    y: float


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
