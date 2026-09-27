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
    run. Deliberately orchestrator-agnostic: `LocalRunner` and a future
    Layer 2 remote/distributed orchestrator both record into the same
    interface, so bookkeeping looks the same regardless of how a pipeline
    was actually run or registered - "however they are orchestrated" is the
    point, not local-only. This is distinct from the `ArtifactStore`: the
    artifact store persists *values*, the metadata store persists *what
    happened* - `record_task_output` stores an artifact key, not the
    artifact itself. `PostgresMetadataStore` requires the optional
    `psycopg` dependency (this project's `postgres` extra), imported lazily
    so the rest of `pipelines` never requires it to be installed.
* `CodeBundler`: Packages a project root directory into a single archive
    and uploads it to a `BaseArtifactStore`, once per pipeline run, so every
    task's remote runtime unpacks the exact same code regardless of which
    machine/container executes it - mirroring Metaflow's "code package".
    Deliberately bundles everything under the root (minus ignored patterns)
    rather than statically analysing which files a specific task needs;
    the third-party equivalent (which packages does a task's runtime need
    installed) remains the `@uv` decorator's job, not this.
* `LocalRunner`: Executes an `AssembledPipeline` locally - runs each
    `AssembledTask`'s function in topological order. Only two kinds of
    value are ever materialized through a `BaseArtifactStore` (a
    `LocalArtifactStore` by default): each pipeline input, once, before any
    task runs; and each task's output, once computed - the same save/load
    mechanism a real, distributed Layer 2 execution will eventually use. No
    task input is ever independently materialized: a `TaskOutput`- or
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
    run's bookkeeping into a `BaseMetadataStore` (a `LocalMetadataStore` by
    default): first, the `AssembledPipeline`'s own structure (via
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
* `Frontend` (`docker/frontend/`): A read-only viewer for browsing what a
    `PostgresMetadataStore`/`S3ArtifactStore` pair holds, as three views: a
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
    (`docker/frontend/backend/`, a thin REST wrapper around the two stores)
    and a React + TypeScript + Vite + Tailwind app (`docker/frontend/web/`)
    that the backend serves as static files - one image, built in two
    Dockerfile stages (Node builds the app, Python serves it). Talks to the
    remote store flavours directly and only those, since a separate
    container has no access to whichever machine ran a `LocalRunner`
    against a local file/SQLite store. Not part of Layer 1's own package
    (`bettmensch_ai.pipelines`) - it's a separate consumer of it, the same
    as any other script would be, just packaged as its own docker image.
    Brought up alongside `postgres`/`minio` by
    `sdk/test/docker-compose/pipelines.docker-compose.yaml` (`make
    pipelines.up`) for local development.

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

`S3ArtifactStore` and `PostgresMetadataStore` need something real to talk
to - `sdk/test/docker-compose/pipelines.docker-compose.yaml` provides a
disposable local Postgres and MinIO for exactly that. Three test tiers
build on it:

* `sdk/test/unit/pipelines/`: no real infrastructure at all - pure logic
    (`LocalArtifactStore`/`LocalMetadataStore` against `tmp_path`,
    `S3ArtifactStore`/`PostgresMetadataStore`'s wiring against mocks).
* `sdk/test/integration/pipelines/`: one store against its real backend
    in isolation (`S3ArtifactStore` against real MinIO,
    `PostgresMetadataStore` against real Postgres).
* `sdk/test/functional/pipelines/`: a full `LocalRunner` pipeline run
    across all 4 valid `BaseArtifactStore` x `BaseMetadataStore`
    combinations (local/local, local/postgres, s3/local, s3/postgres),
    independently re-loading each task's output from the artifact store
    using only the key the metadata store recorded for it - proving the two
    stores actually agree on what happened, not just that `LocalRunner`
    claims they do.

`sdk/test/conftest.py`'s `postgres_dsn`/`s3_config` fixtures probe
reachability once per test session (not per test - each probe takes a
few seconds when unreachable, so per-test would add up fast) and skip,
independently per backend, rather than fail, if the compose stack isn't
up - so `local/local` always runs, and having only one of the two services
up still runs everything that only needs that one.