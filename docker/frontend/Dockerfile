# Read-only viewer for `pipelines` run/registration bookkeeping: a FastAPI
# backend (thin wrapper around PostgresMetadataStore/S3ArtifactStore) serving
# a React+TypeScript+Vite+Tailwind app as static files. Built from the repo
# root as context (needs the root pyproject.toml/uv.lock plus sdk/, not just
# this directory) - see sdk/test/docker-compose/pipelines.docker-compose.yaml's
# `frontend` service (local dev stack) and docker/frontend/docker-compose.yaml
# (versioned build/push, mirroring docker/component's own convention).

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
RUN uv sync --extra postgres --extra frontend --frozen

COPY sdk ./sdk
COPY docker/frontend/backend ./backend
COPY --from=web-build /web/dist ./backend/static

# `bettmensch_ai` isn't pip-installed here, just source-copied (matching how
# pytest.ini's `pythonpath = sdk` already makes it importable for tests
# without an install step, not a Dockerfile-specific shortcut) - so `sdk`
# needs to be on PYTHONPATH explicitly, and `/app` too so `backend` resolves
# as a package (uvicorn, like streamlit before it, doesn't add anything
# useful to sys.path on its own the way `python script.py` does).
ENV PATH="/app/.venv/bin:${PATH}" PYTHONPATH="/app/sdk:/app"

EXPOSE 8080

# python, not curl: avoids an extra apt-get layer just for a healthcheck.
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')" || exit 1

ENTRYPOINT ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
