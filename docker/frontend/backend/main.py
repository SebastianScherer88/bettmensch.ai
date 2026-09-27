"""FastAPI app: the JSON API under `/api`, plus the built React app as static
files for everything else (with an SPA fallback to `index.html` so
client-side routes like `/runs/<id>` survive a direct load/refresh).
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import router

app = FastAPI(title="bettmensch.ai pipelines")
app.include_router(router, prefix="/api")

STATIC_DIR = Path(__file__).parent / "static"

if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str) -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")
