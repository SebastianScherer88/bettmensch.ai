# Accepted Design Decisions

## Layer 1 first draft (2026-09-19)

The following decisions were made while implementing the first draft of
Layer 1 (`sdk/bettmensch_ai/pipelines`) and are proposed here for
record, since they refine details `architecture.md` leaves open rather than
contradicting it.

* **Plain typed functions instead of IO marker classes.** `Task`/`Pipeline`
  functions use ordinary type-annotated python arguments (e.g. `def add(a:
  int, b: int) -> int`) rather than the previous SDK's `InputParameter`/
  `OutputArtifact` marker types. Since Layer 1 is backend-agnostic, the
  Parameter-vs-Artifact distinction is no longer a Layer 1 concern - it is
  decided later, by whichever `Materializer` the assembler resolves for a
  given type.
* **`AssembledPipeline`.** Not named explicitly in `architecture.md`, but
  introduced as the `Pipeline`-level counterpart to `AssembledTask`: the
  `Assembler`'s output is a topologically ordered `AssembledPipeline`
  (tasks + `IOBinding`s + resolved outputs), analogous to how a `Task`
  becomes an `AssembledTask`.
* **`Assembler`, not `Compiler`.** Layer 1's `Pipeline` -> `AssembledPipeline`
  step is named `assemble`/`Assembler`, reserving `compile`/`Compiler`/
  `CompiledPipeline` for Layer 2's backend-specific step (per
  `architecture.md`'s own Layer 2 description). The naming convention is:
  decorate a function -> `Pipeline` (backend-agnostic) -> `assemble()` ->
  `AssembledPipeline` (backend-agnostic) -> a Layer 2 compiler's `compile()`
  -> `CompiledPipeline` (backend-specific).
* **`@pipeline` assembles eagerly by default.** The decorator traces and
  assembles the function immediately, so the decorated name is bound to an
  `AssembledPipeline`, not the intermediate `Pipeline` - pass
  `@pipeline(assemble=False)` to opt out and get the lazy `Pipeline` instead.
  This is safe to do at decoration/import time because assembly never
  depends on concrete input values (every declared parameter is always
  replaced by a `PipelineInput` placeholder, regardless of what a caller
  would eventually pass), only on how task calls wire together. The
  trade-off is that a broken pipeline definition now raises at import time;
  `assemble=False` is the escape hatch when that is undesirable (e.g.
  importing a module just to introspect which pipelines it defines).
* **`TaskOutput`/`PipelineInput`/`PipelineOutput` live in `io_binding.py`,
  not under `task/` or `pipeline/`.** These reference types point at each
  other (a `PipelineOutput`/`IOBinding`'s source can be a `TaskOutput` or a
  `PipelineInput`), so splitting them across the `task` and `pipeline`
  packages would make those two packages depend on each other just for
  these types - and, since importing any submodule of a package runs that
  package's `__init__.py` in full first, a `Task` needing `PipelineInput`
  while `pipeline.io` needed `TaskOutput` becomes a real import cycle, not
  just an ordering nuisance. Keeping this small, dependency-free family of
  types in one shared module avoids the cycle structurally, so `task/` and
  `pipeline/` never need to reach into each other.
* **Pipeline input defaults live on `PipelineInput.default`, task input
  defaults are resolved directly into `AssembledTask.static_inputs`.** These
  are two different mechanisms because `Task` and `Pipeline` treat their own
  arguments differently during tracing: a `Task` call receives concrete
  values immediately (`inspect.Signature.bind` + `apply_defaults()`), so an
  omitted, defaulted argument resolves straight into a literal
  `static_inputs` entry - there is nothing left to remember. A `Pipeline`'s
  own parameters, by contrast, are never given concrete values during
  tracing (every one becomes a `PipelineInput` placeholder regardless of
  what a caller would eventually pass, per the eager-assembly decision
  above), so a pipeline input's default can't be resolved away the same
  way - it has to be carried as data on `PipelineInput` itself
  (`default`, defaulting to the `NO_DEFAULT` sentinel; `.required` is
  `default is NO_DEFAULT`) so it survives into `AssembledPipeline.inputs`
  for a future runtime/Layer 2 compiler to see.
* **`exceptions.py` lives at the top level, not under `assembler/`.** A
  `Task` call missing a required input must raise a typed `AssemblyError`
  (`MissingRequiredInputError`) rather than a bare `TypeError`, for the same
  reason every other assembly-time problem does - but `Task` (`task/`) and
  `Assembler` (`assembler/`) both need to raise these, and `assembler/`
  already depends on `task/` (for `AssembledTask`), so `task/` depending
  back on `assembler/` for its own exceptions would point that dependency
  both ways. Moving the exception hierarchy to a shared, dependency-free
  top-level module (the same fix already used for `io_binding.py`) avoids
  it. `Task._assemble` pre-computes each parameter's required-ness once (no
  default, not `*args`/`**kwargs`) and checks it explicitly against
  `bind_partial()`'s result, rather than relying on `Signature.bind()`'s
  own `TypeError` - that lets the resulting `MissingRequiredInputError`
  report which task was being added to the DAG, which inputs are missing,
  and what was already provided for the others (naming the source task for
  a `TaskOutput`, or "pipeline input `<name>`" for a `PipelineInput`).
* **Consistent reference shape across `TaskInput`/`TaskOutput`/
  `PipelineInput`.** A task-scoped reference names its owning (assembled)
  task plus a slot on it (`task_name` + `input_name`/`output_name`); a
  pipeline-scoped reference just names the slot, since there is only one,
  implicit, pipeline (`name`). `IOBinding` used to spell its task-scoped
  target as two loose fields (`target_task_name`/`target_input_name`) with
  a different prefix than `TaskOutput`'s own (`assembled_task_name`) -
  `TaskInput` now gives `IOBinding.target` the same reference shape as
  `IOBinding.source`, so both sides of a binding are symmetric:
  `IOBinding(target: TaskInput, source: Union[TaskOutput, PipelineInput])`.
* **`IOBinding` models connections only, not literal arguments.** A task
  argument that isn't a reference to a pipeline input or another task's
  output is stored as a static/literal input on the `AssembledTask` directly,
  rather than as an `IOBinding`, since it does not describe a connection.
* **A `Task`/`Pipeline` produces exactly one output - no `NamedTuple`
  special-casing.** An earlier draft exposed one named output per field of
  a `NamedTuple` return type, distinct from the single `"result"` output
  any other return type got. That meant two different code paths through
  tracing, assembly, and (once it existed) `LocalRunner`, and the
  NamedTuple path didn't handle every case consistently across all three
  (assembly-time materializer resolution treated a NamedTuple output's
  *fields* as separately typed; execution didn't). Instead, a task's (and
  a pipeline's) output is always exactly one value, named by the shared
  `DEFAULT_OUTPUT_NAME` ("result") constant in `io_binding.py`, whatever
  its type - a function returning a `NamedTuple` is not treated any
  differently than one returning an `int`. If that value happens to be
  JSON-serializable (a `NamedTuple` is a `tuple`), `JsonMaterializer` will
  happily round-trip it, just as a plain list - it comes back as one,
  since JSON has no way to reconstruct the specific NamedTuple type; that
  is an accepted consequence of not special-casing it, not a bug. A
  `Pipeline`'s function is under the same rule: it must return `None`, a
  single `TaskOutput`, or a single `PipelineInput` (a pass-through) -
  never a dict or NamedTuple of several. Fanning out multiple named values
  is a downstream consumer's job.
* **`AssembledPipeline.outputs` is a tuple, matching `.inputs`, holding at
  most one `PipelineOutput`.** With outputs capped at one, `outputs` could
  have collapsed to a single `Optional[PipelineOutput]` field, but keeping
  it a tuple (like `inputs`) means the two sibling collection-shaped
  attributes on the same dataclass share one container type rather than
  one being a dict and the other a tuple, which was a real inconsistency
  in an earlier draft. `.output` is the ergonomic singular accessor
  (`outputs[0] if outputs else None`).
* **Topological order is ranks (`List[List[str]]`), not a flat list.** A
  flat order picks *some* valid sequence but throws away the information
  that, say, two tasks in it don't depend on each other at all and could
  run concurrently. `Assembler._topological_order` now groups tasks into
  ranks via a Kahn's-algorithm-style layering: a rank is every
  not-yet-placed task whose dependencies are all in earlier ranks: no two
  tasks in the same rank can depend on each other. `AssembledPipeline`
  stores this as `task_ranks: Tuple[Tuple[AssembledTask, ...], ...]`
  rather than a flat `assembled_tasks` tuple, so the parallelism
  information survives into the artifact a runtime would consume; a
  runtime that doesn't care just uses the flattening `tasks` property.
  `LocalRunner` itself still executes rank members sequentially - it
  doesn't parallelize - but the structure is there for a runtime that can.
  Cycle detection falls out of the same algorithm: if an iteration
  produces an empty rank while tasks remain, they must be circularly
  dependent.
* **Materializer resolution: static, dynamic, and metadata-based paths.**
  `BaseMaterializer.supports_type(type_hint)` is used by the `Assembler` to
  resolve materializers ahead of execution from type hints;
  `BaseMaterializer.supports(value)` is a runtime fallback for values
  without a statically resolvable type; `resolve_materializer_from_artifact`
  is a third path for when neither exists at all - retrieving an artifact
  after the fact, with no consuming task or type hint anywhere in sight,
  using only the metadata `save()` always writes alongside the data (see
  below).
* **`save()` always writes a metadata sidecar; subclasses implement `_save`/
  `_load`, not `save`/`load`.** For an artifact to be resolvable after the
  fact, something has to record which materializer produced it, since
  there's no consuming task around to supply a type hint. That recording
  has to happen unconditionally, for every materializer, or the "after the
  fact" story only works for whichever ones remembered to do it. Making
  `BaseMaterializer.save()`/`load()` concrete template methods - calling
  abstract `_save`/`_load` for the subclass-specific part, with `save()`
  always following up with `_write_metadata` - makes this the base class's
  job, not each subclass's, so it can't be forgotten. The metadata schema
  is deliberately minimal and stable: `schema_version`, `materializer` (the
  class's `name`, a plain string - not the live class object, which might
  not even exist by the time someone looks), and `value_type` (for human
  debugging only, not used for resolution). `DefaultMaterializer._save`
  always raises before `_write_metadata` runs, so - correctly - no artifact
  ever claims to be materialized by it.
* **`CodeBundler`, not `DependencyResolver`'s static import analysis.** The
  first-draft `DependencyResolver` walked a task function's AST to find the
  minimal set of local files it transitively imports, then bundled only
  those. That's precise but fragile - it can't see dynamic imports,
  `importlib.import_module`, plugins, or anything else that isn't a plain
  `import`/`from` statement - and it re-derives that set independently for
  every task. Metaflow's "code package" solves the same problem more
  simply: bundle everything under one project root (minus ignored patterns)
  once per run, and ship the identical archive to every task, so whatever
  local module a task's (or a pickled object's) code references is
  importable at the same path everywhere, regardless of which machine runs
  it. `CodeBundler` adopts that: `bundle_and_upload` packages the whole
  root once, keyed by `pipeline_name`/`pipeline_run_id` alone (a fixed,
  reserved `CODE_BUNDLE_TASK_NAME`/`CODE_BUNDLE_ARTIFACT_NAME` for the rest,
  since the bundle isn't scoped to any one task), so it's
  naturally idempotent for the same run and every task shares the one key.
  The efficiency cost (shipping code a
  given task doesn't need) is usually negligible next to a real container
  image's third-party dependency weight. This only ever covers the
  project's own code - third-party package dependencies remain the `@uv`
  decorator's job, not this. `download_and_extract` is the read-side
  counterpart; actually placing an extracted bundle on `sys.path` and
  invoking a task's function from inside it is a task entrypoint's job,
  which doesn't exist yet (`LocalRunner` doesn't need one - it runs
  in-process, using whatever code is already loaded - so bundling only
  matters once a real Layer 2 backend exists to consume it).
* **`DefaultMaterializer` refuses to serialize rather than falling back to
  pickle.** Per the "do not silently fall back to unsafe serialization"
  implementation rule: when no specialised materializer supports a type,
  `DefaultMaterializer.save`/`load` raise `NotImplementedError` rather than
  transparently pickling arbitrary objects.
* **`BaseArtifactStore` + `LocalArtifactStore`, not a single boto3-backed
  `ArtifactStore`.** The first-draft `ArtifactStore` stub was implemented
  directly against S3/boto3, which is a specific backend - a poor fit for
  Layer 1, which must stay backend-agnostic (`architecture.md` already
  called it "a local flavour", which this now reflects literally).
  `BaseArtifactStore` is the abstract interface (mirroring
  `BaseMaterializer`); `LocalArtifactStore` is Layer 1's concrete,
  dependency-light default, persisting artifacts as files under a local
  `root_dir`. An S3-backed flavour belongs to Layer 2, as one of the
  AWS-flavoured backend-support abstractions, alongside things like an
  AWS Batch job definition for an `AssembledTask`.
* **`BaseArtifactStore.key()` vs `.uri()`.** `key()` is a concrete method on
  the base class: it builds the backend-agnostic logical address
  (`pipeline/pipeline_run_id/task/artifact_name` - see below for how this
  segment list evolved) shared by every
  backend. `uri()`, `upload()` and `download()` are abstract - each backend
  maps that logical key onto its own physical storage (a local filesystem
  path vs. a bucket + object key, say) and must not conflate the two, e.g.
  by passing a full `s3://...` URI where a backend's transfer call expects a
  bare key.
* **`key()`'s segments are `str` names for "which pipeline"/"which task",
  `uuid.UUID`s only for what needs fresh, per-invocation uniqueness.** The
  first-draft `key()` typed every segment as a `uuid.UUID`, including
  `pipeline_id`/`component_id` - but those identify *which* pipeline/task,
  which `AssembledPipeline.name`/`AssembledTask.name` already do, and are
  already guaranteed unique (dedup'd per pipeline by
  `PipelineAssemblyContext.generate_unique_name`). Forcing them through a
  UUID either meant re-deriving one from the run each time (defeating the
  point of having a stable identifier at all - `LocalRunner` used to
  generate a fresh random `pipeline_id` on *every* `run()` call, so two
  runs of the "same" `AssembledPipeline` weren't recognisably the same
  pipeline in storage) or hashing the name into a UUID for no benefit
  besides satisfying the type. `key()` took `pipeline: str`/`task: str`
  directly at this point - `LocalRunner` passing `assembled_pipeline.name`/
  `assembled_task.name` straight through - while `pipeline_run_id`/
  `task_run_id` stayed `uuid.UUID`, generated fresh by whatever runner was
  driving execution: a `pipeline_run_id` per `run()` call, and a
  `task_run_id` per task invocation, kept on the reasoning that the
  addressing scheme should already be ready for retries/loops/map-style
  invocations without another breaking change once one of those exists.
  **This `task_run_id` reasoning was later reversed** - see the
  `artifact_name` decision further below, which removes it (and
  `artifact_id`) instead of keeping it "just in case". A pleasant side
  effect that *did* survive that later change: storage paths are
  human-readable (now `my-pipeline/<run>/add-1/<output-name>`), which
  directly helps the after-the-fact inspection story.
* **Backend-specific key joining, via an overridable `_join_key_parts`.**
  Different storage media have different native path conventions - an
  object store's keys are conventionally forward-slash-joined strings, a
  local filesystem should use its own OS-native join (`os.path.join`,
  backslash-separated on Windows). `key()` stays concrete on
  `BaseArtifactStore` (it still owns assembling the ordered segments, so
  that logic isn't duplicated per backend), but delegates the actual
  joining to an abstract
  `_join_key_parts(parts)` each backend implements for itself.
  `LocalArtifactStore` uses `os.path.join`; a future `S3ArtifactStore`
  would use `"/".join`. `pathlib.Path` reads either separator back
  correctly regardless of platform, so this doesn't break anything - it
  just keeps a key's on-disk form idiomatic for whichever backend produced
  it, rather than every backend being forced through one convention that
  only really fits object stores.
* **`BaseArtifactStore.save`/`.load`, not `LocalRunner._save`/`_load`.**
  The "stage a value to a temp file via a materializer, then
  upload/download it (data plus its metadata sidecar)" dance isn't a
  runner concern - any runner, local or a future remote one, needs the
  exact same operation to persist a task input/output, regardless of how
  it decides *which* task to run when. Moving it onto `BaseArtifactStore`
  as concrete `save(materializer, value, key)`/`load(materializer, key)`
  methods (built purely on the existing abstract `upload`/`download` +
  `key()`) means every backend gets it for free, and `LocalRunner` no
  longer needs its own copy at all. This did surface a real circular-import
  trap: `materializers/__init__.py` already imports `BaseArtifactStore`
  (for `resolve_materializer_from_artifact`), so if `BaseArtifactStore`
  reached back into `materializers` for the metadata-suffix logic, the two
  packages would depend on each other - the same shape of cycle already
  hit (and fixed) between `task`/`pipeline` earlier. `artifact_metadata.py`
  (top-level, dependency-free, alongside `io_binding.py`/`exceptions.py`)
  holds `metadata_key`/`METADATA_SUFFIX`/`METADATA_SCHEMA_VERSION` instead,
  so both packages import from it without depending on each other; the
  `BaseMaterializer` type itself is only ever needed for a type hint on
  `save`/`load`, so it's imported under `TYPE_CHECKING`, not at runtime.
* **Only pipeline inputs (once, up front) and task outputs (once, when
  computed) are ever materialized - no task input is independently
  re-saved.** An earlier draft round-tripped *every* task input through the
  store, including literal/static values and `PipelineInput`-bound ones,
  reasoning that real, distributed task-to-task communication should
  always cross the store. That reasoning was sound for values flowing
  *between* tasks, but it was applied one level too broadly: a
  `PipelineInput`-bound argument was independently re-materialized by
  *every* consuming task, under a fresh key each time - so a pipeline
  input consumed by two tasks got saved twice, under two unrelated keys,
  for no reason (there is nothing distributed about "the same input,
  looked at twice"). A static/literal input got the same treatment despite
  never being shared with anything at all. The fix: materialize each
  pipeline input exactly once, before any task runs (`_materialize_pipeline_inputs`),
  and have every consuming task *load* it from that one key; a static
  input is used directly, in memory, and never touches the store. A
  `TaskOutput`-bound input already reused its producer's key rather than
  re-saving (that part was already correct) - it's now handled by the same
  `_load` helper as a pipeline-input read, rather than its own inline
  branch. This didn't need a new "runtime-enriched `IOBinding`" concept:
  the existing `task_output_keys` lookup table, plus one more of the same
  shape for pipeline inputs (`pipeline_input_keys: Dict[str, str]`), is
  already the right amount of indirection - `IOBinding` stays purely
  symbolic, and these plain dicts are the runner's own concrete key
  resolution, built at run time. One knock-on simplification: a
  `TaskOutput` load can just as easily use `resolve_materializer_from_artifact`
  as the producer's own resolved one, so `LocalRunner._load` resolves
  every load's materializer from the artifact's own stored metadata
  uniformly, rather than threading materializer objects around alongside
  keys. (At the time this was written, pipeline inputs had no
  assembly-time-resolved materializer of their own either, and were
  resolved dynamically, from their value, at save time - that gap is
  closed by the next decision below.) A tracked follow-up:
  `Assembler._resolve_task_materializers` still resolves a materializer
  for every task *input* (not just its output), but `LocalRunner` now only
  ever reads the output entry - the input entries are currently unused at
  runtime, though they still provide early, assembly-time validation that
  a declared input type is materializable at all. Left as-is rather than
  trimmed, since removing it is an independent decision about how much to
  prize that early validation.
* **Pipeline input materializers are resolved by the `Assembler`, from the
  pipeline function's own type hints, and persisted on `AssembledPipeline`
  - not resolved dynamically at run time.** The previous decision above
  left a real asymmetry: every task input/output gets its materializer
  resolved statically, at assembly time (catching an unmaterializable
  declared type before the pipeline ever runs), but a pipeline input got
  its materializer resolved *dynamically*, from its actual value, inside
  `LocalRunner` - the one place in the whole design that didn't "resolve
  materializers during compilation where possible". `Pipeline` now carries
  its own `type_hints` (via `get_type_hints`, mirroring `Task`), and the
  `Assembler` resolves a `Dict[str, BaseMaterializer]` from them the same
  way `_resolve_task_materializers` does for a task - just once, for the
  whole pipeline, rather than per-task, since a pipeline input isn't
  scoped to any single task's type hints and so can't reuse theirs. The
  result is persisted as `AssembledPipeline.input_materializers`, mirroring
  `AssembledTask.materializers`. `LocalRunner._materialize_pipeline_input`
  now takes the already-resolved materializer as a parameter instead of
  calling `resolve_materializer_for_value`, so an untyped pipeline
  parameter now falls back to `DefaultMaterializer` (and so fails loudly,
  at save time, rather than silently succeeding via dynamic resolution) -
  the same behaviour an untyped task input/output already had, not a new
  inconsistency.
* **`LocalRunner`'s own `ExecutionError` hierarchy is not a subclass of
  `AssemblyError`.** Assembling a pipeline (validating its graph, resolving
  materializers) and running it are different concerns - `architecture.md`
  and the design principles treat "compilation" and "execution" as
  deliberately separate, so their errors shouldn't share a base class
  either. `MissingPipelineInputError`/`UnknownPipelineInputError` are
  raised when a *run* is given the wrong inputs, which is a runtime
  concern: the same `AssembledPipeline` can be assembled once and run many
  times with different inputs, so this validation could never have
  happened at assembly time.
* **`PydanticJsonMaterializer` reconstructs its model class from two
  separate metadata fields (`model_module`/`model_qualname`), not one
  concatenated string, and via `BaseMaterializer`'s new `_extra_metadata`/
  `from_metadata` hooks rather than a special case in
  `resolve_materializer_from_artifact`.** The materializer previously took
  an optional `model: Type[...]` constructor argument, but nothing ever
  actually passed it: every resolution path (`resolve_materializer_for_type`,
  `resolve_materializer_for_value`, `resolve_materializer_from_artifact`)
  constructed materializers with `materializer_cls()`, so `load()` always
  returned a plain `dict`. `_write_metadata` on `BaseMaterializer` now
  merges in whatever `self._extra_metadata(value)` returns (default: `{}`),
  and `resolve_materializer_from_artifact` reconstructs a materializer via
  `materializer_cls.from_metadata(metadata)` (default: plain `cls()`)
  instead of always calling the bare constructor - two small, generic
  extension points any materializer can use, not a pydantic-specific
  mechanism. `PydanticJsonMaterializer._extra_metadata` records
  `type(value).__module__`/`type(value).__qualname__` - the value's actual
  runtime type, not `self.model` - so the recorded class is correct
  regardless of whether the saving instance happened to have `model`
  configured. They're kept as two fields rather than one
  `f"{module}.{qualname}"` string because a nested class's `__qualname__`
  (e.g. `"Outer.Inner"`) already contains dots, making it impossible to
  tell, from a single dotted string alone, how many trailing components
  belong to the class path versus the module path.
  `from_metadata`/`_resolve_class` re-import the module and walk the
  qualname via `getattr`, degrading to an unconfigured materializer
  (`model=None`, `load()` returns a `dict`) rather than raising, if the
  module isn't importable, a name doesn't resolve, the result isn't a
  pydantic `BaseModel`/`BaseSettings` (sub)class, or the qualname names a
  function-local class (a `<locals>` component - there is no import path to
  such a class at all, bundled code or not; that would need actual
  object-graph serialization, i.e. pickle, not an import). Since
  `LocalRunner._load` already resolves every load via
  `resolve_materializer_from_artifact` (previous decision above), this one
  fix applies uniformly to both in-pipeline execution and after-the-fact
  inspection - no separate fix was needed for the assembly-time-resolved
  path. It still depends on the model's module actually being importable
  in whatever process is deserializing - via ordinary installation for a
  third-party model, or (for this project's own code) a `CodeBundler`
  bundle already extracted onto `sys.path`, which nothing does
  automatically today; see `CodeBundler.download_and_extract`.
* **A `Task` returning a `NamedTuple` or `TypedDict` gets one independently
  named, independently materialized output per field/key; every other
  return type - including a plain `dict` or `tuple` - remains one opaque
  output named `DEFAULT_OUTPUT_NAME` ("result"), unchanged.** This reverses
  the earlier decision recorded above ("each task and pipeline can only
  return one object... a `NamedTuple` is not treated any differently than a
  plain value") for the `NamedTuple`/`TypedDict` case specifically, not for
  return types in general - a real, deliberate reversal, not an extension;
  `docs/architecture.md`'s "no per-field multi-output mechanism" language
  and the tests that encoded it (`test_assemble_treats_named_tuple_return_
  as_a_single_opaque_output`, `test_task_returning_a_named_tuple_still_has_
  a_single_output`, `test_run_treats_named_tuple_output_as_a_single_opaque_
  value`) were rewritten, not extended. Considered making this opt-in (a
  decorator flag, or requiring every task to declare a `NamedTuple`/
  `TypedDict` return type), but rejected both: an opt-in flag adds syntax
  for something already fully determined by the return type annotation the
  `Assembler` already reads, and requiring it universally would force
  boilerplate (a dedicated output class) onto every single-output task -
  by far the common case - for no benefit to tasks that only ever produce
  one thing. A `NamedTuple`/`TypedDict` return type is detected the same
  way the standard library itself would (`issubclass(tp, tuple) and
  hasattr(tp, "_fields")` for a `NamedTuple`; `typing.is_typeddict(tp)` for
  a `TypedDict`), purely from `Task.type_hints["return"]` - no execution of
  the function body required, preserving "resolve materializers during
  compilation where possible" for the multi-output case exactly as it
  already held for the single-output one. A pure runtime, side-effecting
  declaration mechanism (e.g. a Metaflow-`self.x=...`-style `declare_output`
  call from inside the function body) was considered and rejected as the
  *primary* mechanism: `Pipeline.trace()` calls a `Task` symbolically,
  without ever running its function body (`Task._assemble` records the
  invocation and returns a placeholder; the body only runs later, inside
  `LocalRunner._run_task`) - so a pipeline body needs to reference a
  specific output *during tracing*, before the function has run even once,
  which only a statically-declared return shape can support.
  `Task._build_output` reuses the *same* declared `NamedTuple`/`TypedDict`
  class as the shape of that symbolic placeholder (constructing a real
  instance of it, or a plain dict for `TypedDict`, with each
  field/key holding a `TaskOutput` instead of a real value) rather than a
  bespoke `TaskOutputs` wrapper class - this is the same trace-time
  surrogate trick `TaskOutput`/`PipelineInput` already play standing in for
  a concrete value; a type checker sees the declared field type (e.g.
  `TorchModule`) where a `TaskOutput` actually sits, harmless because it's
  confined entirely to pipeline-definition code. `Task.output_name: str`
  generalizes to `Task.output_names: Tuple[str, ...]` (and
  `AssembledTask.output_name` likewise to `output_names`) - length 1 in the
  ordinary case - with `Task.output_type_hints: Dict[str, Any]` resolving
  each output name to *its own* type hint (each field's/key's own
  annotation for a multi-output task, or `{DEFAULT_OUTPUT_NAME: return_type}`
  otherwise) so `Assembler._resolve_task_materializers` resolves each
  output's materializer independently, from its own type, rather than from
  the outer `NamedTuple`/`TypedDict` type as a whole. `TaskOutput`/
  `IOBinding`/`Pipeline._resolve_output` needed no changes at all: a
  multi-output field reference (`outputs.model`) is just a `TaskOutput`
  with an arbitrary `output_name`, exactly like the existing single-output
  case - that machinery already generalized cleanly, it just wasn't being
  exercised with more than one `output_name` per task before now.
  `LocalRunner._run_task` now materializes each of `output_names`
  separately (extracting each field via `getattr`/`dict.__getitem__`
  depending on which of `is_named_tuple_output`/`is_typed_dict_output` the
  task is), under its own key, rather than materializing one combined
  return value. Left open, deliberately not decided here: whether
  materializer resolution should happen at trace/assembly time at all
  (as opposed to at runtime) - a separate, broader question about the
  `Assembler`'s design raised in discussion but not yet settled, so this
  decision proceeds on today's assembly-time resolution without taking a
  position on that question either way.
* **`Pipeline`/`@pipeline` accept a `default_materializer` class, passed
  through to the `Assembler` and used instead of a hardcoded
  `DefaultMaterializer` wherever a task input/output or pipeline input's
  type isn't supported by any of `MATERIALIZER_REGISTRY`'s specialised
  materializers - defaulting to `DefaultMaterializer` itself, so its
  refuse-rather-than-silently-fall-back behaviour is unchanged unless a
  pipeline explicitly opts out.** Considered making `DefaultMaterializer`
  itself attempt pickle/cloudpickle, but rejected it: `DefaultMaterializer`
  is specifically the thing chosen automatically when nothing else
  matches, so if it also silently pickled, any unrecognized type would go
  back to being silently pickled by default - reintroducing exactly the
  risk "do not silently fall back to unsafe serialization" was written to
  rule out. Making the fallback an explicit, per-pipeline choice keeps that
  default intact while still giving a pipeline author a real escape hatch
  when they've deliberately decided the risk is acceptable for their case.
  `resolve_materializer_for_type`/`resolve_materializer_for_value` gained a
  `default_materializer_cls` parameter (still defaulting to
  `DefaultMaterializer`) rather than each caller special-casing the
  fallback itself. Because a pipeline's chosen fallback is only ever
  selected explicitly, not from `MATERIALIZER_REGISTRY`'s automatic
  type/value-based chain, `Pipeline.__init__` registers it into
  `MATERIALIZER_BY_NAME` via the new `register_materializer` (excluded from
  `MATERIALIZER_REGISTRY` itself) - otherwise an artifact it produced would
  record a `materializer` name in its metadata sidecar that
  `resolve_materializer_from_artifact` has no class for. No concrete
  pickle/cloudpickle materializer ships yet; this only adds the seam for a
  pipeline to plug one in.
* **`LocalRunner` reconciles an assembly-time-resolved materializer against
  the actual value about to be saved, rather than resolving materializers
  from type hints alone or from runtime values alone.** Prompted by a real
  question: Python's type hints aren't enforced at runtime, so a task
  declared `-> int` that actually returns something else would, under pure
  type-hint-based resolution, get the *wrong* materializer chosen for it -
  and its `_save()` fails with a confusing, several-layers-removed error
  that doesn't obviously point back to the inaccurate type hint. Considered
  switching to fully dynamic, value-based resolution instead (the codebase
  already has `resolve_materializer_for_value` for exactly this, though
  nothing actually called it from the main save path before this decision),
  but rejected it as the *primary* mechanism: it would make
  `AssembledTask.materializers`/`AssembledPipeline.input_materializers` a
  best-guess rather than a fully-determined plan, undermining "resolve
  materializers during compilation where possible" and complicating Layer
  2's eventual job of compiling an `AssembledPipeline` into a remote
  backend representation ahead of any task actually running anywhere. It
  would also directly reverse the earlier, deliberate fix making pipeline
  input materializers resolve statically at assembly time instead of
  dynamically at run time (see above). Two further observations shaped the
  final shape: first, *loading* was never affected by this question at all
  - `LocalRunner._load` already resolves purely from the artifact's own
  metadata sidecar (`resolve_materializer_from_artifact`), never from a
  consumer's statically-resolved materializer, so accuracy there was never
  in question; the only moment a type hint's accuracy actually matters is
  the first *save*, before any metadata exists to consult instead. Second,
  static resolution's "fail fast" benefit doesn't actually address the
  failure mode in question - it only catches a *declared* type nothing
  supports, which is orthogonal to a type hint that's simply *wrong* (by
  construction, a wrong hint still looks fine to static analysis). Given
  both, the chosen middle ground: keep static, assembly-time resolution as
  the default (cheap, keeps the plan fully determined, unchanged for the
  overwhelmingly common case where the hint is accurate), but at the actual
  `artifact_store.save(...)` call, check `materializer.supports(value)`
  (a method every materializer already implements, previously only
  exercised inside `resolve_materializer_for_value` itself, not from the
  main save path) and re-resolve via `resolve_materializer_for_value` if it
  fails - carrying `AssembledPipeline.default_materializer` (a new field,
  the `Pipeline`'s own default, threaded through from `Assembler.assemble`)
  as the fallback if nothing in the registry matches the actual value
  either, so a pipeline's own fallback choice is honoured even during
  reconciliation, not silently replaced by the plain `DefaultMaterializer`.
  This reconciliation deliberately is not silent: a
  `MaterializerMismatchWarning` (`RuntimeWarning`, not an `ExecutionError` -
  the run itself recovers and continues) is raised, since a type-hint/
  reality mismatch signals either a genuine bug or a type hint that should
  be widened (e.g. to a `Union` or `Any`), and staying quiet about it would
  cut against "do not silently fall back" in spirit even though the
  fallback here is a correction, not a degradation. One residual case is
  accepted as out of scope: a wrong type hint that happens to still be
  supported by the *same* materializer as the real value (e.g. declared
  `int`, actually returns `bool` - both go through `JsonMaterializer`) is
  harmless and needs no reconciliation at all.
* **`BaseArtifactStore.key()`/`.uri()` replace `task_run_id`/`artifact_id`
  with a single `artifact_name: str`, dropping randomness from key
  generation entirely.** `task_run_id` never actually identified "which
  invocation" the way its name claimed: `pipeline_run_id` is already fresh
  per `LocalRunner.run()` call, and an `AssembledTask` currently executes at
  most once per pipeline run, so `task_run_id` was pure entropy for
  disambiguating a task's several outputs from each other - and it wasn't
  even doing that coherently, since `_run_task`'s per-output loop generated
  a *different* `uuid.uuid4()` for each output of the *same* invocation.
  `artifact_name` does the actual disambiguating job, deterministically: a
  task output's own name (unique within its task, via `Task.output_names`)
  or a pipeline input's own name, both already-unique, stable strings the
  caller has in hand - no id generation needed. This directly reverses the
  earlier decision (above) to keep `task_run_id` "ready for retries/loops/
  map-style invocations without another breaking change once one of those
  exists" - reconsidered as premature generalization for a construct that
  doesn't exist today (no loop/fan-out primitive), consistent with "do not
  introduce features or abstractions without a concrete use case"; that
  tradeoff can be revisited specifically if/when such a construct is added,
  rather than paid for now. A useful side effect: a key is now fully
  deterministic given `(pipeline, pipeline_run_id, task, artifact_name)`,
  making retries/overwrites of the same artifact natural and letting an
  artifact's key be computed directly rather than only recovered from a
  runner's own `task_output_keys`/`pipeline_input_keys` bookkeeping.
  `artifact_name` (not `output_name`) is the name at the `BaseArtifactStore`
  level specifically because the store itself doesn't know about tasks or
  outputs as pipeline concepts - `pipeline`/`task` are already just
  caller-supplied labels to it - and the same slot also carries a pipeline
  *input's* name (not an output) when `task` is the reserved
  `PIPELINE_INPUT_TASK_NAME` sentinel, so a name scoped to "output"
  specifically would have been inaccurate for that case. That sentinel
  pattern - already used for pipeline inputs and for `CodeBundler`'s
  `CODE_BUNDLE_TASK_NAME` - now extends uniformly to the 4th segment too
  (`CODE_BUNDLE_ARTIFACT_NAME`, replacing the two `CODE_BUNDLE_TASK_RUN_ID`/
  `CODE_BUNDLE_ARTIFACT_ID` UUID sentinels with one string sentinel): every
  "this isn't really a task's output" case now reuses the same mechanism
  rather than each inventing its own.
* **`BaseMetadataStore`/`LocalMetadataStore`, a new abstraction alongside
  `BaseArtifactStore`/`LocalArtifactStore`, not folded into either the
  artifact store or `LocalRunner` itself.** The artifact store persists
  *values* (a task output's actual serialized bytes); this persists *what
  happened* (a pipeline/task run's status and timing, and which artifact
  key each output landed under) - different enough concerns, with
  different natural query shapes (structured, filterable records vs.
  content-addressed blobs), that conflating them into one interface would
  have made both worse. Mirrors the `BaseArtifactStore`/`LocalArtifactStore`
  split exactly: `BaseMetadataStore` is the abstract, orchestrator-agnostic
  interface; `LocalMetadataStore` is Layer 1's concrete, local default; a
  remote, shared flavour reachable by genuinely distributed Layer 2 workers
  is deferred to Layer 2. Not built as part of `LocalRunner` itself either,
  even though `LocalRunner` is its only caller today, because the explicit
  ask was bookkeeping that works "however [pipeline runs] are orchestrated"
  - a future remote orchestrator needs to write into the *same* store
  through the *same* interface, which isn't possible if the bookkeeping
  logic lives inside one specific runner.
* **`LocalMetadataStore` is SQLite-backed, not JSON files.** Bookkeeping is
  inherently about structured records queried back in different shapes (by
  pipeline name, by run id, by task within a run) - `list_pipeline_runs`,
  `list_task_runs`, `list_task_outputs` - not one blob written once and
  re-read whole, the way an individual artifact's metadata sidecar is.
  Hand-rolling that querying over flat JSON files (or one file per record)
  would mean re-implementing basic indexing/filtering by hand and worrying
  about concurrent-write safety across whatever processes report into it.
  `sqlite3` is part of the standard library, so this adds no new
  dependency, and gives real (if minimal) relational structure - three
  tables (`pipeline_runs`, `task_runs`, `task_outputs`) - for free. Each
  store method opens a fresh connection for its own call rather than
  keeping one open for the store's lifetime: bookkeeping calls are
  infrequent (one per task/pipeline run transition, not a hot path), so
  the simplicity of not managing a long-lived connection outweighs the
  modest per-call connection cost.
* **`LocalRunner` records `RUNNING` before attempting work and
  `SUCCEEDED`/`FAILED` (re-raising) afterwards, for both the whole pipeline
  run and each individual task run - never leaving a record permanently
  stuck at `RUNNING`.** `run()` only starts pipeline-run bookkeeping once
  `_resolve_pipeline_inputs` has already validated the given inputs - a
  `MissingPipelineInputError`/`UnknownPipelineInputError` means the run
  never meaningfully began, so no record is created for it at all, rather
  than immediately recording a run that instantly fails validation.
  Everything after that (materializing pipeline inputs, running every
  task, resolving the pipeline's output) is wrapped in one try/except that
  records `FAILED` and re-raises on any exception, `SUCCEEDED` otherwise;
  `_run_task` does the same at the per-task level, one level down, so a
  failure part-way through a multi-task pipeline leaves an accurate
  picture: the failed task (and the run as a whole) marked `FAILED`, every
  task that completed before it marked `SUCCEEDED`, and any task that never
  got to run simply absent from the record entirely (not a third,
  "never started" status) - since nothing was ever recorded for it.
  `record_task_output` is called immediately after each output's `save()`
  succeeds, per output, so a task with several outputs (a `NamedTuple`/
  `TypedDict`-returning one) that fails partway through its own output loop
  still leaves an accurate record of whichever outputs *did* get saved
  before the failure.
* **Pipeline *registration* bookkeeping was initially deferred, then built
  anyway on explicit request.** The entry directly above this one deferred
  it: no backend orchestrator existed yet to register a pipeline with, so
  there was nothing concrete to validate a schema against. That reasoning
  was explicitly overridden - given a concrete instruction to map out and
  implement the schema regardless, the "no concrete use case yet" objection
  no longer applied in the same way; the request itself supplied the
  concrete use case. See the entries below for the resulting design.
* **`PipelineRegistrationRecord`/`TriggerRecord` are separate dataclasses
  from `PipelineRunRecord`/`TaskRunRecord`, on the same `BaseMetadataStore`
  interface rather than a second interface.** A *run* and a *registration*
  are genuinely different things (one specific execution vs. a standing
  declaration that a pipeline exists on some backend), but both are
  "bookkeeping independent of whatever orchestrates things", the same
  reason `BaseMetadataStore` exists at all - splitting them into two
  interfaces would have meant two stores to construct and pass around
  everywhere for what is, underneath, the same persistence concern.
  `dag_structure`/`pipeline_inputs`/`trigger_config` are plain `Dict[str,
  Any]` parameters, not typed against `AssembledPipeline` or a specific
  backend's trigger shapes - keeping `BaseMetadataStore` dependency-free of
  the rest of `pipelines` (it already takes plain `str`s for
  `record_task_output`, not `TaskOutput` objects, for the same reason), and
  sidestepping the very guessing-at-an-unbuilt-backend's-needs problem the
  earlier deferral worried about: a future Layer 2 compiler shapes an
  `AssembledPipeline` into whatever dict form it needs before calling
  `register_pipeline`, rather than the store dictating that shape upfront.
* **A registration/trigger event is a new row, never an update to an
  existing one - except for `deregistered_at`.** Re-registering a pipeline
  (e.g. deploying a new DAG version) calls `register_pipeline` again, which
  creates a *new* `pipeline_registration_id`, rather than overwriting the
  previous registration's row - preserving registration history, mirroring
  how a fresh `pipeline_run_id` per run (not one row reused across runs) was
  already the pattern for run bookkeeping. `deregister_pipeline`/
  `deregister_trigger` are the one intentional exception: they update
  `deregistered_at` in place on an existing row, since a registration/
  trigger being retired is a state change of that *same* registration/
  trigger, not a new event in its own right. `is_active` is a derived
  property (`deregistered_at is None`) rather than a separately stored
  `status` column, since it's fully determined by that one field and
  storing both would risk them disagreeing.
* **`PostgresMetadataStore`, a new, third `BaseMetadataStore`
  implementation, added alongside `LocalMetadataStore` rather than
  extending it.** `LocalMetadataStore`'s entire reason to be SQLite-backed
  was "Layer 1's local, single-process default, no new dependency"; a
  remote store reachable by genuinely distributed Layer 2 workers is a
  fundamentally different deployment shape requiring a real dependency
  (`psycopg`), so it's a new class implementing the same interface -
  mirroring the `BaseArtifactStore`/`LocalArtifactStore` split exactly
  rather than teaching one class to do both jobs. Schema is the same five
  tables in both backends, but with native types where Postgres actually
  has them and SQLite doesn't: `UUID` instead of `TEXT` for every id column,
  `TIMESTAMPTZ` instead of ISO-string `TEXT` for every timestamp, `JSONB`
  instead of `json.dumps`-encoded `TEXT` for `dag_structure`/
  `pipeline_inputs`/`trigger_config`, plus a `CHECK` constraint on `status`
  and real foreign keys between tables (SQLite's schema has neither - not
  retrofitted here, since that wasn't what was asked and changing
  `LocalMetadataStore`'s established schema is a separate decision). psycopg
  adapts `uuid.UUID`/`datetime`/`dict` to and from these native types
  automatically, so `postgres_metadata_store.py`'s row-building helpers need
  no manual (de)serialization the way `local_metadata_store.py`'s
  equivalents do with `str()`/`json.dumps()`/`.isoformat()`.
* **`psycopg` is an optional dependency (this project's `postgres` extra
  in pyproject.toml), imported lazily inside `PostgresMetadataStore.__init__`
  rather than at module level.** Per "do not introduce new dependencies
  without explaining why": every other user of `pipelines` (anyone using
  only Layer 1's local pipeline assembly) has no reason to need a PostgreSQL
  client library installed. A module-level `import psycopg` in
  `postgres_metadata_store.py` would force it onto everyone anyway, since
  `metadata_store/__init__.py` (imported eagerly by `pipelines/__init__.py`)
  would need to import that module to re-export `PostgresMetadataStore` -
  breaking the *entire* package for anyone without `psycopg` installed, not
  just Postgres-specific functionality. Deferring the `import psycopg` (and
  `from psycopg.types.json import Jsonb`) to inside `__init__`, with a clear
  `ImportError` and install instructions if it fails, means the class
  definition and the rest of `pipelines` remain fully importable without
  it; only actually constructing a `PostgresMetadataStore` requires it to be
  installed. Verified directly: `from bettmensch_ai.pipelines.metadata_
  store import PostgresMetadataStore` succeeds with no `psycopg` installed at
  all, and only `PostgresMetadataStore()` itself raises.
* **`PostgresMetadataStore` opens a fresh connection per method call, and
  recreates schema (`CREATE TABLE IF NOT EXISTS`) on every
  initialization, matching `LocalMetadataStore`'s own choices exactly rather
  than optimizing prematurely for a production deployment that doesn't
  exist yet.** A connection pool (`psycopg_pool`) and versioned, explicit
  migrations (rather than an app auto-creating its own schema on connect)
  are both real concerns for a shared database under sustained load from
  many distributed workers - but neither has a concrete requirement driving
  it yet, so both are called out as likely-eventual upgrades in the class's
  own docstring rather than built speculatively now.
* **`S3ArtifactStore` needed no new dependency, unlike `PostgresMetadataStore`
  - `boto3` was already a hard dependency (pyproject.toml has always listed
  it), so it's imported at module level, not lazily.** Everything else
  mirrors `LocalArtifactStore`/`S3ArtifactStoreConfig`'s established shape
  exactly: no auto-created bucket (provisioning storage is a deployment
  concern, matching `LocalArtifactStore` never creating its own root
  directory's volume), `_join_key_parts` forward-slash-joins (S3's own
  convention, already anticipated in `BaseArtifactStore.key`'s own
  docstring), and credentials default to `None` so real AWS S3 usage falls
  through to boto3's normal credential chain (IAM role, environment,
  `~/.aws/credentials`) rather than this class hardcoding a resolution
  strategy - explicit credentials are only for services like MinIO that
  don't participate in that chain.
* **Docker-compose-based local test infrastructure
  (`sdk/test/docker-compose/pipelines.docker-compose.yaml`: Postgres +
  MinIO) plus a three-tier test split (unit / integration / functional),
  rather than mocking S3/Postgres in "integration" tests or adding a
  Python-side orchestration dependency (e.g. `testcontainers`).** Mocking
  would test this project's own assumptions about S3/Postgres's behaviour,
  not whether `S3ArtifactStore`/`PostgresMetadataStore` actually work
  against the real thing - the entire point of an integration tier.
  `testcontainers` (or similar) was considered for making integration/
  functional tests fully self-managing, but rejected for now: it's a new
  dependency for something a one-line `docker compose up -d` already solves
  simply, and per "do not introduce new dependencies without explaining
  why" that's not enough justification on its own. Verified directly,
  end-to-end, against real containers on this machine: `S3ArtifactStore`
  round-tripped a value through real MinIO, `PostgresMetadataStore` recorded
  and read back real run/registration/trigger data through real Postgres,
  and `LocalRunner` completed real pipeline runs across all 4 store
  combinations, cross-checked by independently reloading each task's output
  from the artifact store using the key the metadata store recorded for it
  (not just trusting `LocalRunner`'s own return value or in-memory state).
  MinIO is pulled from `quay.io/minio/minio`, not Docker Hub's `minio/minio`
  - this environment's own Docker registry mirror allows official
  `library/*` images (`postgres:16-alpine` pulled fine) but denied
  `minio/minio` as a third-party org image; quay.io's copy of the same image
  pulled without issue. `LocalMetadataStore`'s SQLite file and
  `LocalArtifactStore`'s filesystem root need no equivalent service, so
  `local/local` is the one combination that needs no compose stack at all
  and always runs.
* **Reachability probes (`postgres_dsn`/`s3_config` in
  `sdk/test/conftest.py`) are session-scoped, with explicit short timeouts
  on the S3 probe specifically.** Discovered directly while building this:
  function-scoped reachability fixtures repeat their probe for every single
  test that (transitively) needs them - each `psycopg.connect(...,
  connect_timeout=2)` against an unreachable host actually takes ~4s in
  practice (it tries both an IPv6 and an IPv4 resolution of `localhost`,
  each with its own timeout), and boto3's *default* retry/backoff made
  repeated failed `head_bucket` probes progressively slower, not just
  slow once - together, running the full "infra down" test session took
  over a minute and looked like a genuine hang before it was tracked down.
  Fixed two ways: the S3 probe's client now uses an explicit
  `botocore.config.Config(connect_timeout=2, read_timeout=2,
  retries={"max_attempts": 1})` so a single failed probe is fast and
  bounded; and `postgres_dsn`/`postgres_metadata_store`/`s3_config`/
  `s3_artifact_store` all moved to `scope="session"`, so the (now-fast)
  probe still only runs once per backend per test session rather than once
  per test - a real efficiency fix independent of the timeout issue, not
  just a workaround for it, since reachability doesn't change mid-session
  and the returned store objects are stateless enough to share safely
  (tests isolate their own data via a fresh `pipeline_run_id`/
  `unique_key_prefix`, not via a fresh store instance).
* **v1 (the Hera/Argo-based `bettmensch_ai.pipelines`, its Streamlit
  dashboard, its `docker/component` task-runtime images) removed outright,
  and `pipelines_v2` renamed to `pipelines` as the only remaining
  version.** v1's own source was already gone from the working tree
  (uncommitted deletions) before this decision; this made that final by
  also removing what was left pointing at it - the now-orphaned v1 test
  suites (`sdk/test/unit/pipelines`, `sdk/test/integration/pipelines`,
  `sdk/test/k8s`, `sdk/test/unit/server`), `docker/component/`,
  `docker/dashboard/` (replaced by the new frontend below), `sdk/setup.py`
  and `sdk/makefile` (v1's pip/extras-based install, fully superseded by
  `sdk/pipelines.makefile`'s `uv`-based one), and the README's v1 examples/
  sections - per "don't worry about removing v1 features/logic, just do
  it," no attempt was made to port anything v1 could do that `pipelines`
  can't. Initially scoped to just the SDK module and its directly-coupled
  tooling, deliberately leaving the underlying Kubernetes/Argo/Terraform
  platform infrastructure (`kubernetes/`, `infrastructure/terraform/`,
  `data_models/`, `make platform.*`) and its now-broken CI workflows
  (`.github/workflows/*`, all built around the deleted `sdk.test`/
  `docker/component` machinery) alone - reasoning it provisioned a cluster
  rather than encoding v1 pipeline logic itself, and remained a plausible
  target for a future Layer 2 backend. **Superseded on explicit
  instruction**: asked directly to also remove that infrastructure and
  "basically all the old content," so it (and the CI workflows) were
  removed too, and the README rewritten from scratch around only what
  currently exists (`pipelines`, its frontend, the local docker-compose
  stack) rather than keeping sections for infrastructure that's now gone.
  No new CI was written to replace the deleted workflows - out of scope
  for this change, flagged as a follow-up if wanted.
  Once v1 was gone, keeping the "_v2" qualifier on the only
  remaining pipeline module became actively misleading (implying a
  sibling that no longer exists), so `sdk/bettmensch_ai/pipelines_v2` and
  every path/import/make-target/env-var-prefix referencing it were renamed
  to drop the suffix.
* **Frontend (`docker/frontend/`): a read-only Streamlit app, added to the
  same local docker-compose stack as Postgres/MinIO rather than a separate
  one.** Streamlit over a JS framework: it's already this project's
  established choice (the deleted v1 dashboard used it too), and a
  read-only, data-table-and-drilldown UI is exactly its sweet spot -
  avoiding a second frontend toolchain (Node/bundler/separate deploy) for
  what two `st.dataframe`/`st.json`-based pages cover. It talks to
  `PostgresMetadataStore`/`S3ArtifactStore` directly (the same classes any
  other consumer would use), never `Local*` stores - a containerized
  frontend has no access to whichever single machine ran a `LocalRunner`
  against a local file/SQLite store, so only the remote flavours make
  sense here. Added to `pipelines.docker-compose.yaml` (not a separate
  compose file) since it shares that stack's `postgres`/`minio`, is not
  profile-gated like `test` (it's meant to be part of the default `up`,
  the normal way to browse local results), and a new `createbuckets`
  one-shot service ensures its bucket exists first - `S3ArtifactStore`
  deliberately never creates its own bucket, and unlike `test`'s own
  disposable bucket (created by a test fixture), local-dev usage needed
  something to create this one. `createbuckets` reuses the `frontend`
  image with a plain boto3 one-liner rather than the usual `minio/mc`
  client image, discovered the hard way: this environment's Docker
  registry mirror blocks third-party org images on Docker Hub (`minio/
  minio`, `minio/mc` both denied; `library/*` images pull fine) - the same
  restriction `minio/minio` itself already had to route around via
  `quay.io` - and quay.io turned out not to have `mc` pullable either, so
  reusing an image that's already proven to build (boto3 is a hard
  dependency regardless) sidesteps the whole question. `streamlit` is a
  new, `frontend`-only optional extra (`pyproject.toml`), not a hard
  dependency - nothing about `pipelines` itself needs it. Verified
  end-to-end, not just written: built and ran the real stack, created a
  pipeline run from the host via `LocalRunner`, and confirmed the
  frontend's own container-side store connections (service-name-based,
  not the host's forwarded ports) correctly saw that run, its task
  statuses, and could resolve and load its actual output values.
  One real bug caught and fixed along the way: `streamlit run` (a
  console-script entry point) doesn't add anything useful to `sys.path`
  the way `python script.py` does, so `bettmensch_ai` - copied in as
  source, not pip-installed, the same convention `pytest.ini`'s
  `pythonpath = sdk` already relies on - wasn't actually importable at
  first; fixed with an explicit `ENV PYTHONPATH=/app/sdk` in the
  Dockerfile rather than relying on incidental behavior.
  **Superseded on explicit instruction**: rejected outright as looking
  wrong for the job ("python + streamlit is a bad choice here... look at
  metaflow, dagster or prefect... make it slick"), so it was replaced with
  a React + TypeScript + Vite + Tailwind CSS app
  (`docker/frontend/web/`) served by a thin FastAPI backend
  (`docker/frontend/backend/`) - a REST API wrapping the same two stores,
  rather than importing them directly into the UI process. Reasoning for
  the replacement: a hand-rolled component set (status badges, a
  recursive collapsible JSON tree, a sidebar) over a heavy component
  library, for full control over the "slick," Metaflow/Dagster/Prefect-like
  look the instruction asked for, which Streamlit's boxy widget-per-line
  layout can't produce regardless of styling effort. Kept as one docker
  image, not two: a multi-stage Dockerfile builds the React app in a
  `node:20-slim` stage, then copies the built static files into the
  Python/FastAPI stage that actually serves them (`StaticFiles`, plus a
  catch-all route falling back to `index.html` so client-side routes like
  `/runs/<id>` survive a direct load or refresh) - avoiding a second
  deployed container/port for what's still, functionally, one artifact.
  `fastapi`/`uvicorn` replaced `streamlit` in the same `frontend` optional
  extra (`pyproject.toml`) rather than adding a third; the published port
  changed from Streamlit's `8501` to `8080` (no meaning attached to the
  old number, just matching a new stack). The `PYTHONPATH=/app/sdk` fix
  from the Streamlit build carries forward unchanged - `uvicorn` is a
  console-script entry point with the identical `sys.path` limitation -
  plus `/app` added alongside it so `backend` itself resolves as an
  importable package from the image's working directory. The API surface
  is a small, deliberately narrow set of read-only endpoints mirroring the
  UI's own needs (list/get pipeline runs with client-side status
  filtering - the store layer itself has no server-side status filter -
  task runs, task outputs, an artifact preview endpoint, and the
  registration/trigger equivalents), not a general-purpose store API;
  widen it if a real second consumer shows up, not preemptively. Artifact
  preview auto-renders a value only when its sidecar metadata's
  `materializer` field is `"json"` or `"pydantic_json"` (the two
  JSON-producing materializers) - anything else (e.g. `polars_parquet`)
  shows its materializer/type as plain metadata instead of attempting a
  preview, rather than guessing at a rendering for formats the tree
  viewer can't meaningfully represent.
* **A thin `docker-compose.yaml` at the repo root, `include`-ing
  `sdk/test/docker-compose/pipelines.docker-compose.yaml` rather than
  duplicating or relocating it.** `docker compose up` with no `-f` flag is
  the default, expected invocation most people reach for first, and
  `make pipelines.up` alone wasn't a discoverable enough answer to that -
  worth a root-level file rather than only documenting the nested path.
  `include:` (Compose 2.20+) keeps the services defined in exactly one
  place; the root file has none of its own. Verified directly:
  `docker compose up -d` (repo root, no `-f`) built and started the same
  four default-profile services (`postgres`/`minio`/`createbuckets`/
  `frontend`, `test` correctly excluded since it stays profile-gated) as
  `make pipelines.up`, with the frontend again shown to correctly see a
  real `LocalRunner` run's data through it.
* **Added persisted pipeline *assembly* bookkeeping (`PipelineAssemblyRecord`,
  a third concern alongside run and registration bookkeeping) rather than
  deriving an "assembled" state from run/registration history.** Requested
  explicitly, to support a Pipelines view showing whether a pipeline has
  been assembled and/or registered, with real dates - before this change,
  assembly was a purely in-memory, ephemeral step (`Assembler.assemble()`)
  with zero persistence anywhere, so a pipeline that had been assembled but
  never run or registered was invisible to any store-backed query. The
  alternative (treating "assembled" as merely implied by the most recent
  run/registration) was rejected in favor of the fuller option, on explicit
  instruction, with one constraint: it had to integrate seamlessly with
  both `LocalMetadataStore` and `PostgresMetadataStore` for a fully local
  setup, not just the remote flavour. Implemented as `record_pipeline_
  assembly`/`get_pipeline_assembly`/`list_pipeline_assemblies` on
  `BaseMetadataStore`, mirroring `register_pipeline`'s shape exactly; a new
  `bettmensch_ai.pipelines.assembler.serialize_assembled_pipeline` turns a
  real `AssembledPipeline` into the `dag_structure`/`pipeline_inputs` dict
  shape (tasks grouped by topological rank, IO bindings, materializers,
  `@resource`/`@uv` requirements) - the first real producer of that
  previously-unshaped dict, and reusable by a future Layer 2 compiler as a
  starting point. `record_assembly` wraps this with a reuse-if-unchanged
  check (comparing against the most recently recorded assembly for that
  pipeline name) so re-running an unchanged pipeline doesn't insert a
  duplicate row per run - assembly history stays one entry per actual
  structural change. `LocalRunner` calls `record_assembly` automatically at
  the top of `run()`, before `start_pipeline_run` - this is the "seamless"
  part: no `LocalRunner` caller needs to opt into assembly bookkeeping
  explicitly, local or remote store alike, since `LocalRunner` already has
  the `AssembledPipeline` in hand right there. `PipelineRunRecord` gained an
  optional `pipeline_assembly_id`, populated by `LocalRunner` from
  `record_assembly`'s return value, specifically so the Runs view's own DAG
  visualization could reference a run's *exact* structural snapshot rather
  than re-deriving it later by matching on pipeline name (which would go
  stale the moment the pipeline's definition changed after that run) - this
  was identified as necessary infrastructure regardless of the assembly-
  bookkeeping question, since task run records never captured dependency
  edges between tasks at all. Also added `PipelineRegistrationRecord.
  backend_metadata` (a separate plain dict, default `{}`) alongside this
  work, for backend-specific resource references (e.g. a Step Functions
  ARN) the Pipelines view can show for a registered pipeline - kept
  separate from `dag_structure` so a caller doesn't need to invent a
  convention for stuffing backend-only fields into what's meant to stay a
  backend-agnostic structure.
  One real migration issue caught and fixed: both stores' schema setup was
  `CREATE TABLE IF NOT EXISTS` only, which - as expected - left an
  *existing* table's columns untouched, so a pre-existing local SQLite file
  or Postgres database (from before this change) crashed on the first
  `start_pipeline_run` call referencing the new `pipeline_assembly_id`
  column. Fixed with an idempotent migration step on every store init:
  `PostgresMetadataStore` just appends `ALTER TABLE ... ADD COLUMN IF NOT
  EXISTS` statements (Postgres supports this directly); `LocalMetadataStore`
  needed a small `_ensure_column`/`_migrate` helper since SQLite's `ALTER
  TABLE ADD COLUMN` has no `IF NOT EXISTS` clause, so it checks `PRAGMA
  table_info` itself first. Verified against a real, already-running
  Postgres database left over from before this change (not a fresh one) -
  the full test suite (153 tests) passed against it without a manual reset,
  confirming the migration path actually works for an existing deployment,
  not just a fresh one.
* **Rebuilt the frontend's three views (Pipelines, Runs, Artifacts) around
  a hand-rolled SVG DAG component (`DagView`) rather than a graph library
  like React Flow.** Requested explicitly, alongside "a nice modern slick
  theme... a cool logo and icons." Each `dag_structure` already carries a
  topological `rank` per task (from `serialize_assembled_pipeline`), which
  makes layout trivial - column = rank, row = position within rank, no
  force-directed/auto-layout algorithm needed - so a full graph library's
  pan/zoom/auto-layout machinery would have been unused weight for what's
  fundamentally a simple layered DAG at modest task counts, consistent with
  this frontend's existing choice to hand-roll `JsonTree` rather than pull
  in a component library. A registration's `dag_structure` is NOT
  guaranteed to have this shape, though (it's deliberately left an
  unshaped, arbitrary dict with no real Layer 2 compiler populating it
  yet - existing test fixtures use e.g. `{"tasks": ["add"], "edges": []}`,
  plain name strings, no ranks) - `DagView` itself only accepts the
  well-formed shape, so a `normalizeDagStructure` helper defensively fills
  in defaults (`rank: 0`, empty IO bindings) for whatever's missing,
  applied to a registration's `dag_structure` before rendering, while an
  assembly's own (always well-formed, since it's `serialize_assembled_
  pipeline`'s direct output) needs no normalization. A `TaskPanel` slide-
  over is shared between the Pipelines view (backend-agnostic: static
  inputs/outputs, materializers, `@resource`/`@uv` requirements) and the
  Runs view (the same, plus runtime status/timing and materialized output
  previews via a shared `ArtifactPreviewCard`, extracted from what was
  previously an inline component in the old flat run-detail list) - one
  component, richer in the run context rather than two near-duplicates.
  The Runs view falls back to the old flat, expandable task list (no DAG)
  for a run with no `pipeline_assembly_id` (e.g. one recorded before this
  feature existed), rather than failing to render - graceful degradation
  for pre-existing data instead of assuming every run now has one. The old
  standalone Registrations pages were removed outright, folded into the
  new Pipelines view (a registration is now one tab on a pipeline's detail
  page, alongside "Assembled") - including trigger display, preserved
  rather than dropped, since the store/API still fully support it even
  though no real backend orchestrator exists yet to create one. The
  Artifacts view's search has no flat backing query (`BaseMetadataStore`
  has none, by design) - it fans out `list_pipeline_runs` ->
  `list_task_runs` -> `list_task_outputs` server-side, filtering by
  pipeline name and run-date range in Python; acceptable at this project's
  dev-tool scale, and a `include_metadata=false` escape hatch skips the
  additional per-artifact metadata-sidecar read (materializer/type shown in
  the results table) if that fan-out ever needs to be cheaper. Verified
  end-to-end against the real stack: rebuilt the image, ran the full test
  suite against the running Postgres/MinIO to generate real assembly/
  registration/run/trigger data, and confirmed every new endpoint
  (`/pipelines`, `/pipelines/{name}`, `/pipeline-assemblies/*`,
  `/artifacts`) returns exactly the shape the new pages expect - including
  a run's `pipeline_assembly_id` correctly resolving to the assembly that
  produced its actual task graph.

* **Added task source code (via `inspect.getsource`) to `dag_structure`,
  and captured stdout/stderr (plus a traceback on failure) as a new
  `TaskRunRecord.logs` field - both requested explicitly, to make a task's
  detail panel show real code and real run output instead of only
  structural metadata.** Source extraction lives in
  `assembler.serialize_assembled_pipeline` (`_task_source`), wrapped in a
  try/except degrading to `None` for a function with no retrievable source
  (REPL, `exec`) - this is presentation detail, not load-bearing structure,
  so a failure there shouldn't break assembly recording. `LocalRunner._run_
  task` wraps only the task function's own call (not the surrounding
  materialization) in `contextlib.redirect_stdout`/`redirect_stderr` into a
  shared `io.StringIO`, appends `traceback.format_exc()` on exception, and
  passes the captured text to `finish_task_run(..., logs=...)` - `None`
  when nothing was captured, not an empty string, so a task that printed
  nothing doesn't render an empty "logs" section. `finish_task_run` gained
  `logs` as a new, optional, backward-compatible parameter (every existing
  caller is unaffected); `task_runs` gained a `logs` column on both stores,
  covered by the same idempotent migration pattern as the assembly-
  bookkeeping addition. Verified directly: a task with two `print()` calls
  showed both lines; a failing task's captured logs showed its own print
  output followed by the real traceback, both through `LocalRunner` against
  the live Postgres/MinIO stack, not just unit-tested in isolation.
  One test bug caught along the way, unrelated to the feature itself but
  surfaced by it: `test_assembler_recording.py` defined its own top-level
  `add(a, b)` task - a name-and-shape collision with four other test files'
  own `add` fixtures - which broke `inspect.getsource` only when the full
  suite ran together (not in isolation), apparently due to how pytest's
  assertion-rewrite import hook interacts with `inspect.findsource` across
  colliding module-level definitions. Confirmed this was test-collection-
  specific, not a real bug (a plain `python script.py` run resolved source
  correctly every time), and fixed by renaming the fixture to something
  unique rather than touching the shared four-file convention. Separately,
  a second flaky test surfaced during this work: `list_pipeline_assemblies`
  ordered purely by `assembled_at DESC`, which left tie-breaking undefined
  whenever two assemblies landed on the same isoformat-string timestamp (a
  real, if rare, possibility at native call speed) - fixed by adding `rowid
  DESC` as an explicit tiebreaker in `LocalMetadataStore` (SQLite's own
  monotonic insertion order), verified stable across repeated runs.
* **Redesigned the DAG view (top-to-bottom instead of left-to-right, pan-
  by-drag instead of native scroll) and the task detail panel (real input/
  output source chips instead of arrow-joined strings, syntax-highlighted
  source code, badge-based resource/uv requirements, a logs panel), plus
  widened the sidebar into a card-style nav with bigger type.** All
  requested explicitly. Top-to-bottom reads more naturally for a pipeline
  (matches Dagster/Prefect's own convention) and was a straightforward
  layout swap once ranks already existed as the layout's organizing axis -
  rank became the row instead of the column, centering shifted from
  vertical-within-column to horizontal-within-row. Panning is a plain
  pointer-event drag (`onPointerDown`/`onPointerMove` updating a translate
  offset, `setPointerCapture` so a fast drag past the element bounds
  doesn't drop the gesture) rather than reaching for a canvas/graph library
  - consistent with the earlier hand-rolled-`DagView`-over-React-Flow
  decision, since panning alone needs none of a graph library's layout
  engine. Added `prism-react-renderer` as a new, narrowly-scoped dependency
  for the source code display specifically - a real syntax highlighter was
  the explicit ask ("nice design, not just strings"), and hand-rolling one
  wasn't a reasonable alternative the way hand-rolling `DagView`'s layout
  was (a layered DAG's layout is simple geometry; Python tokenization
  isn't). Input sources now render as distinct, color-coded chips by kind -
  a task-output source shows the actual upstream task name and output name
  as two joined segments (sky), a pipeline-input source is its own badge
  (violet), a static value renders inline or via `JsonTree` for a
  structured one (amber) - so the three kinds are visually distinguishable
  at a glance, not just readable as text. The sidebar's three sections
  (Pipelines/Runs/Artifacts) each got a distinct accent color (indigo/
  violet/amber, deliberately different from the running/succeeded/failed
  status palette already in use elsewhere, to avoid implying a status
  meaning that isn't there).
