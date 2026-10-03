# Read-only viewer for `pipelines` run/registration bookkeeping: a FastAPI
# backend (thin wrapper around MetadataClient/S3ArtifactStore) serving a
# React+TypeScript+Vite+Tailwind app as static files. Built from the repo
# root as context (needs the root pyproject.toml/uv.lock plus src/, not just
# this directory) - see docker-compose/pipelines.docker-compose.yaml's
# `frontend` service (local dev stack) and docker/frontend/docker-compose.yaml
# (versioned build/push, mirroring docker/component's own convention).
#
# `docker/frontend/pyproject.toml` is this service's own uv workspace
# member - its own deployment deps (fastapi/uvicorn) live there, not in the
# root pyproject.toml; `uv sync --package` below installs only its scope.

ARG BASE_IMAGE_VERSION=3.11-slim

# --- Stage 1: build the React app -------------------------------------------
FROM node:20-slim AS web-build

WORKDIR /web

COPY docker/frontend/web/package.json docker/frontend/web/package-lock.json* ./
RUN npm install

COPY docker/frontend/web ./
RUN npm run build

# --- Stage 2: FastAPI backend, serving the built app -------------------------
FROM python:${BASE_IMAGE_VERSION}

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY docker/frontend/pyproject.toml docker/frontend/pyproject.toml
COPY src ./src
RUN uv sync --package bettmensch-ai-frontend --frozen --no-dev

COPY docker/frontend/backend ./backend
COPY --from=web-build /web/dist ./backend/static

# `bettmensch_ai` is a real (editable) install now via the uv workspace, so
# only `/app` needs to be on PYTHONPATH - for `backend` to resolve as a
# package (uvicorn, like streamlit before it, doesn't add anything useful
# to sys.path on its own the way `python script.py` does).
ENV PATH="/app/.venv/bin:${PATH}" PYTHONPATH="/app"

EXPOSE 8080

# python, not curl: avoids an extra apt-get layer just for a healthcheck.
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')" || exit 1

ENTRYPOINT ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
