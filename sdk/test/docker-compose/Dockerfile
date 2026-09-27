# Containerized test runner for pipelines: joins the same docker-compose
# network as postgres/minio (see pipelines.docker-compose.yaml), so
# integration/functional tests reach them by service name, not
# localhost:<forwarded-port> - closer to how a real CI runner would see
# them than running pytest from the host.
#
# Built from the repo root as build context (needs the root pyproject.toml/
# uv.lock, not just sdk/) - see pipelines.docker-compose.yaml's `test`
# service and sdk/pipelines.makefile's `pipelines.docker.build`.
ARG BASE_IMAGE_VERSION=3.11-slim
FROM python:${BASE_IMAGE_VERSION}

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependencies before source: this layer only invalidates when
# pyproject.toml/uv.lock actually change, not on every source edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --extra postgres --frozen

# Same relative layout as the host repo, so a pytest path/node-id (e.g.
# `sdk/test/unit/pipelines`) means the same thing whether it's run here
# or on a developer's own machine.
COPY pytest.ini ./
COPY sdk ./sdk

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["pytest", "sdk/test/unit/pipelines", "sdk/test/integration/pipelines", "sdk/test/functional/pipelines", "-q"]
