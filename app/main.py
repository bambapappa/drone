import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.routers import health, stream

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    stream.broadcaster.attach(asyncio.get_running_loop())
    src = stream.default_source()
    if src is not None:
        await asyncio.to_thread(stream.manager.start, src)
    yield
    await asyncio.to_thread(stream.manager.stop)


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(stream.router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/player", include_in_schema=False)
async def player():
    return FileResponse(STATIC_DIR / "player.html")
