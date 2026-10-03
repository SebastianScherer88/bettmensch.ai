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

## :package: Package layout (`src/bettmensch_ai/pipelines/`)

Everything below is Layer 1 - local, backend-agnostic pipeline assembly and
execution. There is no Layer 2 (backend-specific compilers/schedulers) in
the active tree right now; see "Stashed: AWS remote compute" below.

| Module | What it holds |
| --- | --- |
| `task/` | `@task`, `Task`, `AssembledTask`, `@resource`/`@uv`. |
| `pipeline/` | `@pipeline`, `Pipeline`, `AssembledPipeline`. |
| `assembler/` | `Assembler` (traces -> validates -> topologically orders), plus `record_assembly`/`serialize_assembled_pipeline`. |
| `io_binding.py`, `context.py`, `exceptions.py` | Supporting types for a traced pipeline: `TaskOutput`/`PipelineInput`/`IOBinding`, the trace-time recording context, and the `AssemblyError` hierarchy. |
| `materializers/` | `BaseMaterializer` + concrete flavours (`JsonMaterializer`, `PolarsParquetMaterializer`, `PydanticJsonMaterializer`, `DefaultMaterializer`) and resolution helpers (`resolve_materializer_for_type`/`_for_value`/`_from_artifact`). |
| `artifact_store/` | `BaseArtifactStore`, `LocalArtifactStore` (filesystem), `S3ArtifactStore` (S3/MinIO). |
| `metadata_store/` | `BaseMetadataStore`, `LocalMetadataStore` (SQLite), `PostgresMetadataStore` (direct Postgres connection), `RemoteMetadataStore` (HTTP to the metadata service). |
| `client/` | `ArtifactClient`/`MetadataClient` (forward to whichever store they're given or build from env config) and `Client` (bundles one of each). The layer a runtime actually holds - see "Features" below. |
| `code_bundler.py` | `CodeBundler` - packages this project's own code once per run, for remote task runtimes to unpack. |
| `compute/` | `BaseComputeBackend`, `LocalComputeBackend` (the only backend currently active - every `AssembledTask` runs in-process). |
| `runner/` | `LocalRunner`/`run_locally` - executes an `AssembledPipeline`, materializing through a `Client`'s `ArtifactClient`/`MetadataClient`. |

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

`pipelines` (`bettmensch_ai.pipelines`, under `src/`) uses `uv` and the
root `pyproject.toml` as its one and only packaging/dependency mechanism -
there's no separate build/install step for the package itself
(`pytest.ini`'s `pythonpath = src` makes it importable directly).

```bash
make pipelines.install
```

installs its dependencies (`boto3`, `httpx`, plus the optional `postgres`/
`api` extras for `PostgresMetadataStore`/the frontend and metadata
service's FastAPI backends) into a local `uv` venv.

## :whale: Local dev stack (Postgres, MinIO, metadata service, Frontend)

`S3ArtifactStore`/`PostgresMetadataStore` need something real to talk to.
From the repository root:

```bash
docker compose up -d
```

brings up Postgres, MinIO, the metadata service (the only process holding
a Postgres DSN - everything else, including the frontend, talks to it over
HTTP instead), and the read-only frontend viewer, all wired together
(there's a `docker-compose.yaml` at the repo root for exactly this - it
`include`s `docker-compose/pipelines.docker-compose.yaml`, so the services
are defined in one place). Equivalently: `make pipelines.up`.

Then open http://localhost:8080, and point your own script's
`S3ArtifactStore`/`PostgresMetadataStore` at `localhost:9000`/`localhost:5433`
- see the frontend's own home page for a copy-pasteable example, or
"Features" below. Tear it down with `docker compose down -v` (or
`make pipelines.down`).

## :wrench: Running tests

Three tiers, under `tests/{unit,integration,functional}/pipelines/`:

- **`unit`** - no real infrastructure at all: pure logic against `tmp_path`
  (`LocalArtifactStore`/`LocalMetadataStore`) and mocked wiring for every
  remote-backed flavour (`S3ArtifactStore`, `PostgresMetadataStore`,
  `RemoteMetadataStore`) plus the `client/` package
  (`test_client.py`: `ArtifactClient`/`MetadataClient`/`Client` build the
  right default store from config and forward every call correctly).
  Always runs, with or without the containers below.
- **`integration`** - one store/client against its real backend in
  isolation: `S3ArtifactStore`/`ArtifactClient` against real MinIO,
  `PostgresMetadataStore` against real Postgres, and `RemoteMetadataStore`
  against a real instance of the metadata service's FastAPI app (started
  in a background thread of the test process itself, backed by the real
  test Postgres - see `test_remote_metadata_store_integration.py`'s own
  docstring for why that's a background uvicorn thread rather than
  `httpx.ASGITransport`).
- **`functional`** - a full `LocalRunner` pipeline run, end to end: every
  valid `ArtifactStore` x `MetadataStore` combination
  (`test_store_combinations_e2e.py`), plus a run via
  `LocalRunner(ArtifactClient, MetadataClient)` against the real,
  docker-compose-*deployed* metadata service and MinIO
  (`test_metadata_service_e2e.py`), independently re-fetched both through
  the client and via a raw `httpx` call straight at the service's own API.

`tests/conftest.py`'s `postgres_dsn`/`s3_config`/`metadata_service_url`
fixtures each probe reachability once per session and **skip** (not fail)
the tests that need them if the corresponding container isn't up - so
`SUITE=unit` never needs the stack below, and bringing up only some of
`postgres`/`minio`/`metadata-service` still runs everything that only
needs those. `integration`/`functional` are also `pytest.mark.integration`/
`.functional`-marked (see `pytest.ini`), if you'd rather filter with `-m`
than by directory.

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
built image, in the same docker network as Postgres/MinIO/the metadata
service, closer to how CI would see it (also brings the containers up and
builds the image for you):

```bash
make pipelines.test.docker SUITE=all
```

See `pipelines.makefile` for the full set of targets/variables.

# Features

## :twisted_rightwards_arrows: `pipelines`

### Overview

`pipelines` lets you decorate plain python functions with `@task`/
`@pipeline` to assemble a DAG - no orchestration platform required to
define or run one locally. A task produces one output per declared name (a
single opaque one by default, or one per field/key if it returns a
`NamedTuple`/`TypedDict`). `LocalRunner` executes an assembled pipeline
in-process, materializing every task output through an `ArtifactClient`
and recording run bookkeeping through a `MetadataClient` - each a thin
forwarding layer in front of a pluggable store: `LocalArtifactStore`/
`LocalMetadataStore` (filesystem/SQLite) by default, or `S3ArtifactStore`
and either `PostgresMetadataStore` (direct connection) or
`RemoteMetadataStore` (HTTP, via the metadata service) for a shared setup
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
from bettmensch_ai.pipelines.client import ArtifactClient, MetadataClient
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

LocalRunner(ArtifactClient(artifact_store), MetadataClient(metadata_store)).run(
    a_plus_b_plus_c, a=3, b=2
)
```

Then refresh the frontend (below) to see it show up.

Equivalently, go through the metadata service instead of connecting to
Postgres directly - the same path the frontend itself uses - by pointing
`MetadataClient` at it instead:

```python
import os
os.environ["BETTMENSCH_AI_METADATA_CLIENT_BACKEND"] = "remote"
os.environ["BETTMENSCH_AI_METADATA_SERVICE_BASE_URL"] = "http://localhost:8081/api"

from bettmensch_ai.pipelines.client import ArtifactClient, MetadataClient

LocalRunner(ArtifactClient(artifact_store), MetadataClient()).run(
    a_plus_b_plus_c, a=3, b=2
)
```

See `tests/unit/pipelines`, `tests/integration/pipelines`, and
`tests/functional/pipelines` for many more worked examples, including
multi-output tasks and every valid `ArtifactStore`/`MetadataStore`
combination.

## :bar_chart: Frontend

A small, read-only React + TypeScript + Tailwind app (`docker/frontend/web/`)
served by a thin FastAPI backend (`docker/frontend/backend/`) for inspecting
whatever's already recorded - metadata via the metadata service (over
HTTP, not a direct Postgres connection), artifacts directly from
`S3ArtifactStore` - it doesn't run, register, or delete anything itself.
Three views:

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

## :inbox_tray: Stashed: AWS remote compute

`stash/aws-remote-compute/` holds a previously working, AWS-verified
Layer 2: `aws_batch()`/`aws_lambda()` task placement,
`StepFunctionsCompiler`/`CompiledPipeline`, `RegisteredPipeline`/
`RemoteRunner`, the `docker/task-runtime/` images, and the corresponding
Pulumi IAM roles/ECR repos/Batch compute environment. It was moved out of
the active tree (not deleted) to clear room for the `Client`/metadata
service redesign described above - every piece of it constructed a raw
`S3ArtifactStore`/`PostgresMetadataStore` directly, and keeping three
remote-compute layers compatible with each step of that redesign would
have meant updating all of them in lockstep with an abstraction that
wasn't settled yet.

The redesign has since landed (this is the `client/` package and
`docker/metadata-service/` above), so re-integrating the stash - pointing
it at `ArtifactClient`/`MetadataClient` instead of a raw store - is now
the next real step for Layer 2, not a reason to keep it stashed further.
See `stash/aws-remote-compute/README.md` for exactly what moved, where
each file came from, and how to bring it back; `docs/design-decisions.md`
for why it happened; and note that the AWS sandbox account used while this
was last verified appears to reset periodically, so treat anything already
provisioned there as gone until re-checked.

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
- [httpx](https://www.python-httpx.org/): HTTP client - backs
  `RemoteMetadataStore`'s calls to the metadata service.
- [uv](https://docs.astral.sh/uv/): Python packaging and dependency
  management.
