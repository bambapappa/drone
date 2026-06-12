from fastapi import FastAPI
from app.core.config import settings
from app.routers import health

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    debug=settings.debug,
)

app.include_router(health.router)


@app.get("/")
async def root():
    return {"app": settings.app_name, "version": settings.version}
