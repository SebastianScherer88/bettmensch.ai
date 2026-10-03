"""FastAPI app: the metadata service. Pure JSON API under `/api` - unlike
the frontend, this has no UI to serve, just `BaseMetadataStore` over HTTP.
"""

from fastapi import FastAPI

from .routes import router

app = FastAPI(title="bettmensch.ai metadata service")
app.include_router(router, prefix="/api")
