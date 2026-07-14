"""Review UI + annotation REST API for the offline analysis tool.

Thin client over the artifact (architecture report §2.4/§2.5). Reads from
the sidecar store that the analysis CLI writes; writes only to the separate
append-only annotations log (bookmarks + screenshots in Phase 2). Never
touches the engine — no detection/tracking/identity/event code is imported
here, by design (interface rule 2: the UI touches only the artifact + the
annotation REST API, never the engine).

Components:
  - main.py            FastAPI app, mounts static + wires routers
  - annotations.py     Append-only annotation log (separate from AI tables)
  - routes.py          REST endpoints (runs, events, tracklets, bookmarks…)
  - config.py          Settings (output dir, video dir, port)
  - static/            HTML/CSS/JS thin client (HTML5 <video> + overlay canvas)

Run natively:     uvicorn review.main:app --port 8001
Run via compose:  docker compose -f docker-compose.yml -f docker-compose.offline.yml up review
"""

__all__ = ["main"]
