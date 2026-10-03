# The metadata service: a FastAPI wrapper around a real `PostgresMetadataStore`
# - the only process in the stack that holds a Postgres DSN. Every other
# process talks to it over HTTP via `RemoteMetadataStore`. Built from the
# repo root as context (needs the root pyproject.toml/uv.lock plus src/, not
# just this directory) - see docker-compose/pipelines.docker-compose.yaml's
# `metadata-service` service.

ARG BASE_IMAGE_VERSION=3.11-slim

FROM python:${BASE_IMAGE_VERSION}

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --extra postgres --extra api --frozen

COPY src ./src
COPY docker/metadata-service/backend ./backend

# `bettmensch_ai` is source-copied, not pip-installed - matching
# docker/frontend/Dockerfile's own PYTHONPATH convention.
ENV PATH="/app/.venv/bin:${PATH}" PYTHONPATH="/app/src:/app"

EXPOSE 8081

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8081/api/health')" || exit 1

ENTRYPOINT ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8081"]
