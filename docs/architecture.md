# Overview

The orchestrator has two main layers:

## Layer 1: Local, backend agnostic pipeline assembly

The internal pipeline definition which is agnostic of remote execution and 
orchestration backends. This layer is the user's development interface and lets
them annotate python functions with appropriate decorators to assemble DAGs.
Compiling that backend-agnostic assembly into a backend-specific
representation is Layer 2's job, not Layer 1's.

Core abstractions in this layer are:

* `Task`: User-facing task definition. Produces one output per name in
    `output_names`: a single one, named `DEFAULT_OUTPUT_NAME` ("result"),
    for an ordinary return type - a plain value, a `dict`, a plain `tuple`,
    a pydantic `BaseModel`, whatever; or one independently named,
    independently materialized output per field/key if the return type is
    annotated as a `NamedTuple` or `TypedDict`. Detection is purely
    type-based, not opt-in: any `NamedTuple`/`TypedDict` return annotation
    triggers it. Calling such a task while a `Pipeline` is being traced
    returns an instance of that same declared class (fields holding a
    `TaskOutput` each) or a plain `dict` (keys holding a `TaskOutput` each)
    instead of a bare `TaskOutput`, so a pipeline body references a
    specific output the same way it would read the real value
    (`outputs.field` / `outputs["key"]`), autocomplete and all - see
    `Task._build_output`. Associated decorators include:
    - `@resource` for memory, cpu, and gpu requirements, 
    - `@uv` for uv-managed dependencies to be made available in the task's runtime
* `Pipeline`: User-facing pipeline definition. Also produces exactly one
    output: its function must return `None`, a single `TaskOutput`, or a
    single `PipelineInput` pass-through. `default_materializer` lets a
    pipeline opt into a non-refusing fallback (e.g. a pickle/cloudpickle-
    based materializer) for any input/output type none of the specialised
    materializers support; it defaults to `DefaultMaterializer`, so the
    "refuse rather than silently fall back to something unsafe" behaviour
    is unchanged unless a pipeline explicitly opts out of it, per-pipeline.
* `AssembledTask`: Resolved, executable task representation of a task being 
    referenced in a pipeline definition.
* `AssembledPipeline`: The `Assembler`'s output - a validated execution plan
    for a `Pipeline`. Its `task_ranks` groups `AssembledTask`s into
    topologically ordered ranks (`Tuple[Tuple[AssembledTask, ...], ...]`):
    every task in a rank depends only on tasks in earlier ranks, never on
    another task in the same rank, so a parallelism-capable runtime can run
    a whole rank concurrently (the flattened `tasks` property is available
    when that doesn't matter). `inputs` and `outputs` are both tuples, kept
    structurally consistent with each other; `outputs` holds at most one
    `PipelineOutput` (`.output` is the singular accessor), matching a
    `Pipeline`'s one-output rule. `input_materializers` resolves each
    declared input's `Materializer` from the pipeline function's own type
    hints, the same static, assembly-time resolution a `Task`'s
    inputs/output get - a pipeline input isn't scoped to any single task's
    type hints, so it can't reuse theirs. The backend-agnostic artifact
    Layer 2 compiles from.
* `IOBinding`: Describes connections between task outputs and inputs.
* `Assembler`: Resolves task references, validates `IOBinding`s, resolves
    `Materializer`s, detects cycles, and groups tasks into the topologically
    ordered ranks that make up the `AssembledPipeline`. Kept independent of
    runtime scheduling.
* `Materializer`: Serializes and deserializes task inputs and outputs and helps submit and retrieve them to and from the `ArtifactStore`. Facilitates the exchange of i/o objects across pipeline tasks both in a local and all of the remote execution contexts. `BaseMaterializer` is the abstract interface; concrete flavours (`JsonMaterializer`, `PolarsParquetMaterializer`, `PydanticJsonMaterializer`, `DefaultMaterializer`) are resolved per type by the `Assembler`, falling back to a pipeline's `default_materializer` (itself `DefaultMaterializer` unless overridden) rather than hardcoding `DefaultMaterializer` directly - this is what lets a pipeline opt into e.g. pickling for otherwise-unsupported types, deliberately and per-pipeline, without weakening the default. Every `save()` also writes a small metadata sidecar recording the materializer's stable `name` (not the live class) plus whatever extra fields a subclass's `_extra_metadata` hook contributes, so `resolve_materializer_from_artifact` can pick the right materializer back up - via the matching `from_metadata` hook - later using only what is in the `ArtifactStore`, no consuming task or declared type hint required. This is what makes an artifact inspectable after the fact (e.g. in a notebook, long after the run that produced it): `PydanticJsonMaterializer`, for instance, records the exact `__module__`/`__qualname__` of the value's runtime class (as two separate fields, since a nested class's `__qualname__` already contains dots that would make a single concatenated string ambiguous to split back apart) and dynamically re-imports it on load, falling back to a plain `dict` if the class can't be resolved (module not importable, renamed, or function-local) rather than raising. A materializer only ever selected explicitly (like a pipeline's `default_materializer` override) is registered into the by-name lookup via `register_materializer` so it, too, stays resolvable after the fact even though it's excluded from the automatic type/value-based resolution chain.
* `ArtifactStore`: A dedicated artifact persistence layer that pipeline
executions and task invocations can submit inputs and outputs to and
retrieve from. `BaseArtifactStore` is the abstract interface;
`LocalArtifactStore` is Layer 1's concrete, backend-agnostic default,
persisting artifacts to the local filesystem; `S3ArtifactStore` is a
remote, shared, Layer 2-flavoured implementation, persisting to S3 (or an
S3-compatible service, e.g. MinIO for local testing) - reachable by every
worker in a genuinely distributed run, unlike `LocalArtifactStore`. Built
using `boto3`, already a hard dependency of this project, so no new one was
needed. `key()` addresses an artifact by `pipeline`/`task`/`artifact_name` (all
stable, human-readable names) plus `pipeline_run_id` (fresh per run) - a
task currently executes at most once per pipeline run, so no separate
per-invocation id is needed on top of that, and the key is fully
deterministic given these four values;
backends supply their own `_join_key_parts` (forward-slash for an object
store, `os.path.join` for a local filesystem). `save(materializer, value,
key)`/`load(materializer, key)` are concrete conveniences built on top of
`upload`/`download`, so every backend gets the materializer-aware
save/load path for free.
* `MetadataStore`: Bookkeeping of pipeline runs and the task runs within
    them - status (`RUNNING`/`SUCCEEDED`/`FAILED`), timing, and which
    artifact key each task output was materialized under - plus bookkeeping
    of pipelines that have been *assembled* (a dated snapshot of a pipeline's
    backend-agnostic DAG structure, independent of whether it's ever run or
    registered) and pipelines *registered* with a remote backend orchestrator
    (their DAG structure, declared inputs, backend-specific metadata, and any
    schedules/triggers registered against them) - independent of whatever
    actually orchestrates execution or registration. `BaseMetadataStore` is
    the abstract interface;
    `LocalMetadataStore` is Layer 1's concrete, local, SQLite-backed
    default; `PostgresMetadataStore` is a remote, shared, Layer 2-flavoured
    implementation, reachable by every worker in a genuinely distributed
    run, talking to Postgres directly; `RemoteMetadataStore` is a third
    flavour that talks HTTP to the metadata service
    (`docker/metadata-service/`) instead - see below. Deliberately
    orchestrator-agnostic: `LocalRunner` and a future Layer 2 remote/
    distributed orchestrator both record into the same interface, so
    bookkeeping looks the same regardless of how a pipeline was actually run
    or registered - "however they are orchestrated" is the point, not
    local-only. This is distinct from the `ArtifactStore`: the artifact
    store persists *values*, the metadata store persists *what happened* -
    `record_task_output` stores an artifact key, not the artifact itself.
    `PostgresMetadataStore` requires the optional `psycopg` dependency (this
    project's `postgres` extra), imported lazily so the rest of `pipelines`
    never requires it to be installed.
* `Client` (`bettmensch_ai.pipelines.client`): the layer every runtime (a
    `LocalRunner` script today; later, remote compute) actually holds,
    instead of a raw store directly. `ArtifactClient`/`MetadataClient` each
    mirror `BaseArtifactStore`/`BaseMetadataStore`'s own public methods
    exactly (explicit one-line forwarding methods, same names/signatures) -
    deliberately *not* a subclass of the store they forward to, so a client
    and a store can never be type-confused for one another, even though the
    vocabulary is identical. Each builds its own default store from
    `ArtifactClientConfig`/`MetadataClientConfig` (env-driven:
    `local`/`s3` for artifacts, `local`/`remote` for metadata) when none is
    given explicitly - `local` stays exactly Layer 1's own default
    (`LocalArtifactStore`/`LocalMetadataStore`); `s3`/`remote` select
    `S3ArtifactStore`/`RemoteMetadataStore`. `Client` bundles one of each
    (`.artifact_storage_client`, `.metadata_client`) as the one object a
    runtime needs to attach. Utilities that take a store
    (`resolve_materializer_from_artifact`, `CodeBundler.bundle_and_upload`/
    `download_and_extract`, `assembler.record_assembly`/
    `record_assembly_from_dicts`) accept either a raw store or the
    corresponding client, since a client forwards every call identically.
* `Metadata service` (`docker/metadata-service/`): a FastAPI service
    exposing the *entire* `BaseMetadataStore` surface (read and write, unlike
    the read-only frontend API) over REST, backed by a real
    `PostgresMetadataStore` - the only process in the whole stack that holds
    a Postgres DSN. Every other metadata consumer (a `LocalRunner` script
    configured for the `remote` backend, the frontend's backend, later a
    remote compute task) talks to it via `RemoteMetadataStore`/
    `MetadataClient` instead of connecting to Postgres directly - mirroring
    Metaflow's own metadata-service architecture. Brought up alongside
    `postgres`/`minio`/`frontend` by
    `docker-compose/pipelines.docker-compose.yaml`.
* `CodeBundler`: Packages a project root directory into a single archive
    and uploads it to a `BaseArtifactStore`, once per pipeline run, so every
    task's remote runtime unpacks the exact same code regardless of which
    machine/container executes it - mirroring Metaflow's "code package".
    Deliberately bundles everything under the root (minus ignored patterns)
    rather than statically analysing which files a specific task needs;
    the third-party equivalent (which packages does a task's runtime need
    installed) remains the `@uv` decorator's job, not this.
* `LocalRunner`: Executes an `AssembledPipeline` locally - runs each
    `AssembledTask`'s function in topological order. Takes an
    `ArtifactClient`/`MetadataClient` (each defaulting to a fresh one, local
    backend, if omitted) rather than a raw store directly - see `Client`
    above. Only two kinds of value are ever materialized through the
    `ArtifactClient`: each pipeline input, once, before any task runs; and
    each task's output, once computed - the same save/load mechanism a
    real, distributed Layer 2 execution will eventually use. No task input
    is ever independently materialized: a `TaskOutput`- or
    `PipelineInput`-bound input is *loaded* from whichever of those was
    already materialized, never re-saved, and a static/literal input is
    used directly, in memory. Before saving a value, `_reconcile_materializer`
    checks that the materializer resolved at assembly time (from a declared
    type hint) actually `supports()` the real value about to be saved - type
    hints aren't enforced at runtime, so a task/pipeline input can produce
    something its declared type didn't promise. On a mismatch it re-resolves
    from the actual value instead (via `resolve_materializer_for_value`,
    falling back to the pipeline's own `default_materializer` same as
    assembly time would) and raises a `MaterializerMismatchWarning` -
    recoverable, so the run continues, but surfaced loudly since it signals
    either a bug or a type hint that should be widened. Also records this
    run's bookkeeping into the `MetadataClient`: first, the
    `AssembledPipeline`'s own structure (via
    `assembler.record_assembly` - a no-op if it's unchanged since the last
    recorded assembly of this pipeline, so assembly history stays one entry
    per actual change rather than one per run), referenced from the run
    record itself so a run's DAG visualization stays accurate even if the
    pipeline's definition changes later; then the pipeline run's own status,
    each task run's status (plus whatever it wrote to stdout/stderr while it
    ran, captured via `contextlib.redirect_stdout`/`redirect_stderr` around
    just the task function's own call, with a traceback appended on
    failure - `TaskRunRecord.logs`), and each task output's artifact key,
    wrapping both the whole run and each individual task in a try/except
    that records `FAILED` and re-raises on an exception, `SUCCEEDED`
    otherwise. Deliberately independent of the
    `Assembler`, per "separate compilation from execution": assembling and
    validating a pipeline is one concern, running an already-valid one is
    another. Not the "runtime scheduler" mentioned in the project overview -
    that is Layer 2's job for remote/distributed execution; `LocalRunner` is
    a local-only, single-process convenience for development and testing.
* `Frontend` (`docker/frontend/`): A read-only viewer for browsing what the
    metadata service/`S3ArtifactStore` hold, as three views: a
    **Pipelines** view (every pipeline name the store knows of, whether
    assembled and/or registered, with an interactive DAG visualization per
    assembly/registration and a click-through per-task detail panel -
    backend-agnostic for an assembly, plus `backend_metadata`/triggers for a
    registration); a **Runs** view (pipeline runs filterable by status, each
    rendered as the same DAG - via the run's own referenced
    `PipelineAssemblyRecord`, colored by live task status - with a per-task
    panel adding runtime timing and materialized output previews,
    JSON/Pydantic artifacts auto-rendered as an interactive tree); and an
    **Artifacts** view (every task output searchable by pipeline and run
    date, independent of drilling through a specific run). Split into a
    FastAPI backend
    (`docker/frontend/backend/`, a thin REST wrapper) and a React +
    TypeScript + Vite + Tailwind app (`docker/frontend/web/`) that the
    backend serves as static files - one image, built in two Dockerfile
    stages (Node builds the app, Python serves it). Metadata is read
    through a `MetadataClient` configured for the `remote` backend (talking
    to the metadata service over HTTP, like any other remote metadata
    consumer - this backend holds no Postgres connection of its own, unlike
    before the metadata service existed); artifacts are still read directly
    from a real `S3ArtifactStore`, since there was never a proxying concern
    on that side (S3 already has its own secure, IAM-scoped direct-access
    model) - see `get_metadata_client`/`get_artifact_store` in
    `docker/frontend/backend/stores.py`. A separate container has no access
    to whichever machine ran a `LocalRunner` against a local file/SQLite
    store either way. Not part of Layer 1's own package
    (`bettmensch_ai.pipelines`) - it's a separate consumer of it, the same
    as any other script would be, just packaged as its own docker image.
    Brought up alongside `postgres`/`minio`/`metadata-service` by
    `docker-compose/pipelines.docker-compose.yaml` (`make
    pipelines.up`) for local development.
* `AWS infrastructure` (`infrastructure/aws/`): a basic, dev-grade Pulumi
    (Python) stack provisioning genuine AWS counterparts of the local
    docker-compose stack - an S3 bucket for `S3ArtifactStore`, an RDS
    Postgres instance for `PostgresMetadataStore`, IAM roles for the
    frontend's ECS task, an ECR repository for the frontend image, and an
    ECS service running the frontend. See `infrastructure/aws/README.md`
    for provisioning/pushing images, and `docs/design-decisions.md` for
    this stack's explicit "basic" scope cuts (default VPC/no NAT, no ALB,
    `0.0.0.0/0`-reachable by default).

    AWS Batch/Lambda/Step Functions remote compute - `aws_batch()`/
    `aws_lambda()`, `StepFunctionsCompiler`/`CompiledPipeline`,
    `RegisteredPipeline`/`RemoteRunner`, `docker/task-runtime/`'s two
    Dockerfiles, and the corresponding IAM roles/ECR repos/Batch compute
    environment this stack used to provision for them - have all been
    stashed (not deleted): the artifact/metadata `Client` abstractions and
    the metadata service above are that redesign, now complete;
    re-integrating this stash (updating it to go through
    `ArtifactClient`/`MetadataClient` rather than a raw store) is tracked as
    separate future work. See `stash/aws-remote-compute/README.md`.

## Layer 2: Backend specific compilers, orchestration, status management and schedulers/event triggers

The compiling of the backend agnostic internal state into a backend specific
representation which can then be submitted to said backend for execution/scheduling/event based
invocations.

This layer will contain utilities (backend-specific compilers) to map
`AssembledPipeline`/`AssembledTask` onto a backend-specific `CompiledPipeline`
and its constituent counterparts, e.g.

### AWS Stepfunctions

AssembledPipeline -> State Machine definition (a CompiledPipeline)
AssembledTask -> AWS Batch job definition
BaseArtifactStore -> S3ArtifactStore, already implemented (an AWS-flavoured
    `BaseArtifactStore`)
BaseMetadataStore -> PostgresMetadataStore, already implemented (or another
    remote, shared `MetadataStore` flavour), reachable by every distributed
    worker in a run

### Pipeline assembly bookkeeping

`BaseMetadataStore` also covers pipelines that have been *assembled* - a
dated snapshot of an `AssembledPipeline`'s backend-agnostic structure,
independent of whether it's ever run or registered. `record_pipeline_
assembly`/`get_pipeline_assembly`/`list_pipeline_assemblies` record a
`PipelineAssemblyRecord` (`dag_structure`, `pipeline_inputs`,
`assembled_at`) per *distinct* assembly - re-assembling a pipeline whose
definition hasn't changed reuses the existing record's id rather than
inserting a duplicate, so assembly history stays one entry per actual
change, not one per run. `assembler.serialize_assembled_pipeline` turns a
real `AssembledPipeline` into this `dag_structure`/`pipeline_inputs` shape
(tasks grouped by topological rank, their IO bindings - static values,
pipeline inputs, or upstream task outputs - materializers, `@resource`/
`@uv` requirements, and the task function's own source text via
`inspect.getsource` (`None` if unavailable, e.g. a REPL-defined function -
a presentation detail, not load-bearing structure); declared inputs'
required/default/materializer); this
is the same backend-agnostic structure a Layer 2 compiler would also start
from before adding backend-specific content. `assembler.record_assembly`
combines serialization with the reuse-if-unchanged check, and is what
`LocalRunner` calls automatically before every run (see above) - assembly
bookkeeping needs no explicit action from a `LocalRunner` user, local or
remote store alike, though it can also be called standalone to record a
pipeline that's been assembled but not (yet) run. `PipelineRunRecord`
carries an optional `pipeline_assembly_id` pointing at the assembly it
actually executed, populated by `start_pipeline_run`'s caller (`LocalRunner`)
rather than re-derived later by matching on pipeline name - the latter would
go stale the moment a pipeline's definition changes after the run.

### Pipeline registration bookkeeping

`BaseMetadataStore` also covers pipelines that have been *registered* with a
remote backend orchestrator (as opposed to a specific *run* of one) -
`register_pipeline`/`deregister_pipeline`/`get_pipeline_registration`/
`list_pipeline_registrations` record a `PipelineRegistrationRecord` (DAG
structure, declared pipeline inputs, backend-specific metadata,
registration/deregistration time) per registration event, not one row
updated in place, so registration history is preserved across
re-registrations of the "same" pipeline; `register_
trigger`/`deregister_trigger`/`list_triggers` record a `TriggerRecord`
(a backend-defined `trigger_type` plus a free-form `trigger_config` dict)
against a specific registration. Implemented on both `LocalMetadataStore`
and `PostgresMetadataStore` already, even though there is no real backend
orchestrator yet to actually call `register_pipeline` from - a Layer 2
compiler (like the Stepfunctions one above) is what would call these once
it exists. `dag_structure`/`pipeline_inputs`/`trigger_config` are
deliberately plain, JSON-able dicts rather than typed against
`AssembledPipeline` or a specific backend's trigger shapes - a concrete
schema for either would mean guessing at what a not-yet-built backend
compiler actually needs. `backend_metadata` (also a plain, JSON-able dict,
defaulting to `{}`) is kept as its own field rather than folded into
`dag_structure`, specifically for backend-specific resource references
(e.g. a Step Functions state machine ARN) that a caller shouldn't need to
invent a `dag_structure` convention to carry.

## Testing remote store backends

`S3ArtifactStore`, `PostgresMetadataStore`, and the metadata service need
something real to talk to - `docker-compose/pipelines.docker-compose.yaml`
provides a disposable local Postgres, MinIO, and metadata service for
exactly that. Three test tiers build on it:

* `tests/unit/pipelines/`: no real infrastructure at all - pure logic
    (`LocalArtifactStore`/`LocalMetadataStore` against `tmp_path`,
    `S3ArtifactStore`/`PostgresMetadataStore`/`RemoteMetadataStore`'s wiring
    against mocks, and `ArtifactClient`/`MetadataClient`/`Client` building
    the right default store from config and forwarding calls correctly -
    `test_remote_metadata_store.py`/`test_client.py`).
* `tests/integration/pipelines/`: one store/client against its real backend
    in isolation (`S3ArtifactStore` against real MinIO,
    `PostgresMetadataStore` against real Postgres, `ArtifactClient`
    configured for `s3` against real MinIO, and `RemoteMetadataStore`
    against a real instance of the metadata service's FastAPI app running
    in a background thread of the test process - itself backed by real
    Postgres - `test_remote_metadata_store_integration.py`/
    `test_artifact_client_integration.py`).
* `tests/functional/pipelines/`: a full `LocalRunner` pipeline run
    across all 4 valid `BaseArtifactStore` x `BaseMetadataStore`
    combinations (local/local, local/postgres, s3/local, s3/postgres),
    independently re-loading each task's output from the artifact store
    using only the key the metadata store recorded for it - proving the two
    stores actually agree on what happened, not just that `LocalRunner`
    claims they do - plus a `LocalRunner(ArtifactClient, MetadataClient)`
    run against the real, docker-compose-*deployed* metadata service (not
    the integration tier's in-process one) and real MinIO, independently
    re-fetched both via `MetadataClient` and via one raw `httpx` call
    directly against the service's own API
    (`test_metadata_service_e2e.py`) - proving the full
    `client -> service -> Postgres` path end to end.

`tests/conftest.py`'s `postgres_dsn`/`s3_config`/`metadata_service_url`
fixtures probe reachability once per test session (not per test - each
probe takes a few seconds when unreachable, so per-test would add up fast)
and skip, independently per backend, rather than fail, if the compose stack
isn't up - so `local/local` always runs, and having only some of the
services up still runs everything that only needs those.

A fourth tier existed briefly, `aws`-marked, exercising `aws_batch()`/
`aws_lambda()` ad-hoc execution and the full `StepFunctionsCompiler.compile
-> register -> RemoteRunner.run -> deregister` lifecycle against a
genuinely provisioned AWS stack - the one place in this project's test
suite where `boto3` wasn't mocked at all. It's been stashed along with the
remote-compute code it tested - see `stash/aws-remote-compute/README.md`.