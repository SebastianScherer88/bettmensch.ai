# Containerized test runner for pipelines: joins the same docker-compose
# network as postgres/minio (see pipelines.docker-compose.yaml), so
# integration/functional tests reach them by service name, not
# localhost:<forwarded-port> - closer to how a real CI runner would see
# them than running pytest from the host.
#
# Built from the repo root as build context (needs the root pyproject.toml/
# uv.lock, not just src/) - see pipelines.docker-compose.yaml's `test`
# service and pipelines.makefile's `pipelines.docker.build`.
ARG BASE_IMAGE_VERSION=3.11-slim
FROM python:${BASE_IMAGE_VERSION}

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependencies before source: this layer only invalidates when
# pyproject.toml/uv.lock actually change, not on every source edit.
COPY pyproject.toml uv.lock ./
# `api` extra (fastapi/uvicorn): test_remote_metadata_store_integration.py
# runs a real instance of the metadata service's FastAPI app in a
# background thread of the test process itself.
RUN uv sync --extra postgres --extra api --frozen

# Same relative layout as the host repo, so a pytest path/node-id (e.g.
# `tests/unit/pipelines`) means the same thing whether it's run here
# or on a developer's own machine.
COPY pytest.ini ./
COPY src ./src
COPY tests ./tests
# Only the metadata service's own backend code - not its Dockerfile - at
# the same relative path the host repo has it, for the same reason.
COPY docker/metadata-service/backend ./docker/metadata-service/backend

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["pytest", "tests/unit/pipelines", "tests/integration/pipelines", "tests/functional/pipelines", "-q"]
