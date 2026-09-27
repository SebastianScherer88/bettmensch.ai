# :hotel: Welcome to Bettmensch.AI

![bettmensch.ai logo](image/logo_transparent.png)

Bettmensch.AI's `pipelines` is a Python framework for authoring and running
data/ML pipelines: decorate plain functions with `@task`/`@pipeline` to
assemble a DAG, then run it locally with `LocalRunner`, persisting task
outputs and run metadata through pluggable, swappable backends (local files/
SQLite by default, or S3 + PostgreSQL for a shared setup). A small, read-only
React frontend lets you browse what's been run.

See `docs/architecture.md` for the full design and `docs/design-decisions.md`
for the reasoning behind it.

# Setup

## :window: Windows / VSCode (Dev Container)

All `make`/`docker` commands below assume a Unix-like shell with `make`,
`docker`, and `uv` on the `PATH` - none of which Windows has natively. If
you're on Windows, open this repository in VSCode with the
[Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
extension installed, then run **"Dev Containers: Reopen in Container"**
(command palette). This builds a Linux devcontainer (`.devcontainer/`) with
`make`/`uv` installed, and wires up `docker`/`docker compose` inside it to
talk to your host's Docker Desktop daemon directly (no nested
virtualization) - so every command below runs the same way it would on a
native Linux machine or in CI.

## :snake: Install

`pipelines` (`bettmensch_ai.pipelines`, under `sdk/`) uses `uv` and the
root `pyproject.toml` as its one and only packaging/dependency mechanism -
there's no separate build/install step for the package itself
(`pytest.ini`'s `pythonpath = sdk` makes it importable directly).

```bash
make pipelines.install
```

installs its dependencies (`boto3`, plus the optional `postgres`/`frontend`
extras for `PostgresMetadataStore`/the frontend's FastAPI backend) into a
local `uv` venv.

## :whale: Local dev stack (Postgres, MinIO, Frontend)

`S3ArtifactStore`/`PostgresMetadataStore` need something real to talk to.
From the repository root:

```bash
docker compose up -d
```

brings up Postgres, MinIO, and the read-only frontend viewer, all wired
together (there's a `docker-compose.yaml` at the repo root for exactly
this - it `include`s `sdk/test/docker-compose/pipelines.docker-compose.yaml`,
so the services are defined in one place). Equivalently: `make pipelines.up`.

Then open http://localhost:8080, and point your own script's
`S3ArtifactStore`/`PostgresMetadataStore` at `localhost:9000`/`localhost:5433`
- see the frontend's own home page for a copy-pasteable example, or
"Features" below. Tear it down with `docker compose down -v` (or
`make pipelines.down`).

## :wrench: Running tests

```bash
make pipelines.test SUITE=unit          # no infrastructure needed
make pipelines.test SUITE=integration   # needs the containers above
make pipelines.test SUITE=functional    # needs the containers above
make pipelines.test SUITE=all           # all three
```

Narrow a run to one file or one test case with `MODULE`/`TEST_CASE`:

```bash
make pipelines.test SUITE=integration MODULE=test_s3_artifact_store_integration.py
make pipelines.test SUITE=unit MODULE=test_metadata_store.py TEST_CASE=test_list_triggers_is_empty_when_none_registered
```

Or run the whole suite fully containerized - pytest itself runs inside a
built image, in the same docker network as Postgres/MinIO, closer to how CI
would see it (also brings the containers up and builds the image for you):

```bash
make pipelines.test.docker SUITE=all
```

See `sdk/pipelines.makefile` for the full set of targets/variables.

# Features

## :twisted_rightwards_arrows: `pipelines`

### Overview

`pipelines` lets you decorate plain python functions with `@task`/
`@pipeline` to assemble a DAG - no orchestration platform required to
define or run one locally. A task produces one output per declared name (a
single opaque one by default, or one per field/key if it returns a
`NamedTuple`/`TypedDict`). `LocalRunner` executes an assembled pipeline
in-process, materializing every task output through a `BaseArtifactStore`
and recording run bookkeeping through a `BaseMetadataStore` - both
pluggable: `LocalArtifactStore`/`LocalMetadataStore` (filesystem/SQLite) by
default, or `S3ArtifactStore`/`PostgresMetadataStore` for a shared setup
(see "Local dev stack" above).

### Example

```python
from bettmensch_ai.pipelines.task import task
from bettmensch_ai.pipelines.pipeline import pipeline
from bettmensch_ai.pipelines.runner import LocalRunner

@task
def add(a: int, b: int) -> int:
    return a + b

@pipeline
def a_plus_b_plus_c(a: int, b: int, c: int = 2) -> int:
    a_plus_b = add(a, b)
    return add(a_plus_b, c)

result = LocalRunner().run(a_plus_b_plus_c, a=3, b=2)
print(result)  # 7
```

To persist to the shared Postgres/MinIO stack instead of local defaults:

```python
from bettmensch_ai.pipelines.artifact_store import S3ArtifactStore, S3ArtifactStoreConfig
from bettmensch_ai.pipelines.metadata_store import PostgresMetadataStore, PostgresMetadataStoreConfig

artifact_store = S3ArtifactStore(S3ArtifactStoreConfig(
    bucket="bettmensch-ai-artifacts",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="bettmensch_ai",
    aws_secret_access_key="bettmensch_ai_secret",
))
metadata_store = PostgresMetadataStore(PostgresMetadataStoreConfig(
    dsn="postgresql://bettmensch_ai:bettmensch_ai@localhost:5433/bettmensch_ai_metadata",
))

LocalRunner(artifact_store, metadata_store).run(a_plus_b_plus_c, a=3, b=2)
```

Then refresh the frontend (below) to see it show up.

See `sdk/test/unit/pipelines`, `sdk/test/integration/pipelines`, and
`sdk/test/functional/pipelines` for many more worked examples, including
multi-output tasks and every valid `ArtifactStore`/`MetadataStore`
combination.

## :bar_chart: Frontend

A small, read-only React + TypeScript + Tailwind app (`docker/frontend/web/`)
served by a thin FastAPI backend (`docker/frontend/backend/`) for inspecting
whatever a `PostgresMetadataStore`/`S3ArtifactStore` pair already holds - it
doesn't run, register, or delete anything itself. Three views:

- **Pipelines**: every pipeline name the store knows of, filterable by
  whether it's been assembled and/or registered, each with an interactive
  DAG - backend-agnostic for an assembly, plus backend-specific metadata and
  triggers for a registration - and a click-through panel per task (inputs/
  outputs, materializers, `@resource`/`@uv` requirements).
- **Runs**: every recorded pipeline run, filterable by status, rendered as
  the same DAG (colored by each task's live status) via the run's own
  snapshotted assembly; clicking a task adds its runtime timing and
  materialized output(s) - JSON and Pydantic-model artifacts render as an
  interactive, collapsible tree straight from the artifact store, other
  formats (e.g. Parquet) show their materializer/type as metadata instead.
- **Artifacts**: every task output across every run, searchable by pipeline
  and run date, independent of drilling through a specific run.

See "Local dev stack" above to run it. To build/push a standalone,
versioned image instead:

```bash
make frontend.build DOCKER_ACCOUNT=your-account
make frontend.push DOCKER_ACCOUNT=your-account
```

See `docker/frontend/makefile` for the full set of targets.

# Credits

This project makes liberal use of various great open source projects:
- [FastAPI](https://fastapi.tiangolo.com/): the `pipelines` frontend's API
  backend.
- [React](https://react.dev/) + [Vite](https://vitejs.dev/) +
  [Tailwind CSS](https://tailwindcss.com/): the `pipelines` frontend's UI.
- [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html):
  AWS SDK for Python - backs `S3ArtifactStore` (and works against
  S3-compatible services like MinIO for local development).
- [psycopg](https://www.psycopg.org/): PostgreSQL adapter for Python -
  backs `PostgresMetadataStore`.
- [uv](https://docs.astral.sh/uv/): Python packaging and dependency
  management.
