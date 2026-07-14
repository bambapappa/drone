"""FastAPI app for the review UI.

Mounts the static UI under / and the REST API under /api. The UI is a thin
HTML/CSS/JS client served as static files; playback uses the native HTML5
<video> element (no WS streaming, no server-side frame pushing — report §2.5).

Run natively:     uvicorn review.main:app --port 8001
Run via compose:  docker compose -f docker-compose.yml -f docker-compose.offline.yml up review
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from review.routes import router as api_router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Drönare · Granskningsvy",
    version="0.1.0",
    description="Tunn granskningsklient över analys-arkivet (offline-verktyg, fas 2).",
)

app.include_router(api_router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the single-page review UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok"}
