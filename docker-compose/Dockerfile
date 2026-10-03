# Containerized test runner for pipelines: joins the same docker-compose
# network as postgres/minio (see pipelines.docker-compose.yaml), so
# integration/functional tests reach them by service name, not
# localhost:<forwarded-port> - closer to how a real CI runner would see
# them than running pytest from the host.
#
# Built from the repo root as build context (needs the root pyproject.toml/
# uv.lock/src/, not just tests/) - see pipelines.docker-compose.yaml's `test`
# service and pipelines.makefile's `pipelines.docker.build`.
ARG BASE_IMAGE_VERSION=3.11-slim
FROM python:${BASE_IMAGE_VERSION}

RUN pip install --no-cache-dir uv

WORKDIR /app

# `bettmensch-ai` is a real (editable) package now via the uv workspace, so
# `uv sync` below needs `src/` present to build it - this layer no longer
# invalidates on pyproject.toml/uv.lock changes alone, only on source edits
# too, but editable builds are cheap (no compilation), so this costs little.
COPY pyproject.toml uv.lock ./
COPY src ./src
# Root `dev` group (not an `--extra` flag) covers everything this image's
# test run needs: `psycopg`/`fastapi`/`uvicorn` included - the latter two
# specifically back test_remote_metadata_store_integration.py, which runs a
# real instance of the metadata service's FastAPI app in a background
# thread of the test process itself.
RUN uv sync --frozen

# Same relative layout as the host repo, so a pytest path/node-id (e.g.
# `tests/unit/pipelines`) means the same thing whether it's run here
# or on a developer's own machine.
COPY pytest.ini ./
COPY tests ./tests
# Only the metadata service's own backend code - not its Dockerfile - at
# the same relative path the host repo has it, for the same reason.
COPY docker/metadata-service/backend ./docker/metadata-service/backend

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["pytest", "tests/unit/pipelines", "tests/integration/pipelines", "tests/functional/pipelines", "-q"]
