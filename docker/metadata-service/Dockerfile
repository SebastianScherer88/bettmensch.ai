# The metadata service: a FastAPI wrapper around a real `PostgresMetadataStore`
# - the only process in the stack that holds a Postgres DSN. Every other
# process talks to it over HTTP via `RemoteMetadataStore`. Built from the
# repo root as context (needs the root pyproject.toml/uv.lock plus src/, not
# just this directory) - see docker-compose/pipelines.docker-compose.yaml's
# `metadata-service` service.
#
# `docker/metadata-service/pyproject.toml` is this service's own uv
# workspace member - its own deployment deps (fastapi/uvicorn, plus
# `bettmensch.ai[postgres]` for psycopg) live there, not in the root
# pyproject.toml; `uv sync --package` below installs only its scope.

ARG BASE_IMAGE_VERSION=3.11-slim

FROM python:${BASE_IMAGE_VERSION}

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY docker/metadata-service/pyproject.toml docker/metadata-service/pyproject.toml
COPY src ./src
RUN uv sync --package bettmensch-ai-metadata-service --frozen --no-dev

COPY docker/metadata-service/backend ./backend

# `bettmensch_ai` is a real (editable) install now via the uv workspace, so
# only `/app` needs to be on PYTHONPATH - matching
# docker/frontend/Dockerfile's own convention.
ENV PATH="/app/.venv/bin:${PATH}" PYTHONPATH="/app"

EXPOSE 8081

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8081/api/health')" || exit 1

ENTRYPOINT ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8081"]
