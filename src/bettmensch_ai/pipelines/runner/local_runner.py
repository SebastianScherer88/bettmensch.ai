"""`LocalRunner`: executes an `AssembledPipeline` in-process, dispatching
each task to wherever its `compute_backend` places it.
"""

from __future__ import annotations

import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type, Union

from ..assembler.recording import record_assembly
from ..client import ArtifactClient, MetadataClient
from ..code_bundler import CodeBundler
from ..io_binding import IOBinding, TaskOutput
from ..materializers import resolve_materializer_for_value, resolve_materializer_from_artifact
from ..metadata_store import RunStatus
from ..pipeline.assembled_pipeline import AssembledPipeline
from ..task.assembled_task import AssembledTask
from .exceptions import (
    MaterializerMismatchWarning,
    MissingPipelineInputError,
    UnknownPipelineInputError,
)
from .task_execution import get_captured_logs

if TYPE_CHECKING:
    from ..materializers.base_materializer import BaseMaterializer

_TaskOutputKey = Tuple[str, str]

# Reserved "task" sentinel for a pipeline input's own materialized artifact
# key - can never collide with a real (dasherized) task name, the same way
# `code_bundler.CODE_BUNDLE_TASK_NAME` can't.
PIPELINE_INPUT_TASK_NAME = "__pipeline_input__"


class LocalRunner:
    """Executes an `AssembledPipeline` locally: a local *orchestrator* that
    lines up each rank's tasks in topological order and dispatches each one
    to wherever its own `compute_backend` places it - in-process
    (`LocalComputeBackend`, the default) or on remote compute
    (`AwsBatchComputeBackend`/`AwsLambdaComputeBackend`) - while keeping all
    bookkeeping centralized here regardless of where a task actually ran.
    `LocalRunner` never dispatches an entire *pipeline's* execution to a
    remote orchestrator, only individual tasks within one it still
    orchestrates itself; invoking an already-registered pipeline's own
    remote execution (e.g. a Step Functions state machine running the whole
    DAG) is `RemoteRunner`'s job, a separate entry point.

    Only two kinds of value are ever materialized (saved) to the artifact
    store: each of the pipeline's own inputs, exactly once, before any task
    runs; and each of a task's output(s), once computed - a multi-output
    (`NamedTuple`/`TypedDict`-returning) task's fields/keys are each
    materialized independently, under their own key, rather than as one
    combined blob. A task input bound to a `TaskOutput` or `PipelineInput`
    is *loaded* from whichever of those an `IOBinding` points it at - never
    independently re-saved - and a static/literal input is used directly,
    in memory, never touching the store at all. Tasks within a rank are
    independent of each other (per `AssembledPipeline.task_ranks`) but are
    still run sequentially here; `LocalRunner` does not itself parallelize
    them, even across remote-placed tasks.

    Deliberately independent of the `Assembler`: running an already-valid
    `AssembledPipeline` is a separate concern from assembling and
    validating one. Returns the pipeline's single output value directly (or
    `None` if it declares none) - a pipeline, like a `Task`, produces
    exactly one output.

    Also records this run's bookkeeping via a `MetadataClient`: first,
    `assembled_pipeline`'s current structure (via `assembler.record_assembly`
    - a no-op if it's unchanged since the last recorded assembly of this
    pipeline), then the pipeline run's own status (`RUNNING` once inputs
    validate, then `SUCCEEDED`/`FAILED`), each task run's status (plus
    whatever it captured on stdout/stderr, wherever it actually ran -
    `TaskRunRecord.logs`), and each task output's artifact key. This is a
    distinct concern from materialization - the artifact store persists
    *values*, the metadata store persists *what happened* - and
    `LocalRunner` is only one of (eventually) several orchestrators expected
    to write into the same metadata store, local or remote, hence keeping
    the two separate.
    """

    def __init__(
        self,
        artifact_client: Optional[ArtifactClient] = None,
        metadata_client: Optional[MetadataClient] = None,
        project_root: Optional[Union[str, Path]] = None,
    ):
        """Initializes the runner.

        Args:
            artifact_client: The client to materialize pipeline inputs and
                task outputs through. Defaults to a fresh `ArtifactClient()`
                (itself defaulting to a local backend) if omitted.
            metadata_client: The client to record this run's bookkeeping
                (status, timing, task output keys) into. Defaults to a
                fresh `MetadataClient()` (itself defaulting to a local
                backend) if omitted.
            project_root: The directory `CodeBundler` bundles if this run
                has at least one non-local-placed task (nothing is bundled
                for an all-local run). Defaults to the current working
                directory - the natural "project root" for a script being
                run directly - if omitted.
        """

        self.artifact_client = artifact_client or ArtifactClient()
        self.metadata_client = metadata_client or MetadataClient()
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()

    def run(self, assembled_pipeline: AssembledPipeline, **pipeline_inputs: Any) -> Any:
        """Runs `assembled_pipeline` to completion.

        Args:
            assembled_pipeline: The pipeline to run.
            **pipeline_inputs: Values for the pipeline's declared inputs.
                Any input with a default may be omitted.

        Returns:
            The pipeline's single output value, or `None` if it declares
            none.

        Raises:
            MissingPipelineInputError: If a required input is omitted.
            UnknownPipelineInputError: If a value is given for an input the
                pipeline does not declare.
        """

        resolved_inputs = self._resolve_pipeline_inputs(
            assembled_pipeline, pipeline_inputs
        )

        pipeline_assembly_id = record_assembly(self.metadata_client, assembled_pipeline)

        pipeline_run_id = uuid.uuid4()
        self.metadata_client.start_pipeline_run(
            assembled_pipeline.name, pipeline_run_id, pipeline_assembly_id
        )

        try:
            code_bundle_key = self._maybe_upload_code_bundle(
                assembled_pipeline, pipeline_run_id
            )
            pipeline_input_keys = self._materialize_pipeline_inputs(
                assembled_pipeline, pipeline_run_id, resolved_inputs
            )
            task_output_keys: Dict[_TaskOutputKey, str] = {}

            for rank in assembled_pipeline.task_ranks:
                for assembled_task in rank:
                    self._run_task(
                        assembled_pipeline,
                        assembled_task,
                        pipeline_input_keys,
                        task_output_keys,
                        pipeline_run_id,
                        code_bundle_key,
                    )

            result = (
                None
                if assembled_pipeline.output is None
                else self._resolve_output_value(
                    assembled_pipeline, pipeline_input_keys, task_output_keys
                )
            )
        except Exception:
            self.metadata_client.finish_pipeline_run(pipeline_run_id, RunStatus.FAILED)
            raise

        self.metadata_client.finish_pipeline_run(pipeline_run_id, RunStatus.SUCCEEDED)

        return result

    def _maybe_upload_code_bundle(
        self, assembled_pipeline: AssembledPipeline, pipeline_run_id: uuid.UUID
    ) -> Optional[str]:
        """Uploads this run's code bundle once, only if at least one task
        needs it.

        Args:
            assembled_pipeline: The pipeline being run.
            pipeline_run_id: This run's id.

        Returns:
            The code bundle's artifact key, or `None` if every task in
            `assembled_pipeline` runs on a backend that doesn't need one
            (e.g. an all-local pipeline never pays for a bundle it has no
            use for).
        """

        needs_bundle = any(
            task.compute_backend.requires_code_bundle
            for task in assembled_pipeline.tasks
        )
        if not needs_bundle:
            return None

        return CodeBundler(self.project_root).bundle_and_upload(
            self.artifact_client, assembled_pipeline.name, pipeline_run_id
        )

    def _resolve_pipeline_inputs(
        self, assembled_pipeline: AssembledPipeline, pipeline_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validates and resolves the values a `run()` call was given
        against `assembled_pipeline`'s declared inputs.

        Args:
            assembled_pipeline: The pipeline being run.
            pipeline_inputs: The raw values `run()` was called with.

        Returns:
            A mapping from every declared input's name to its resolved
            value (the given one, or its default if omitted).

        Raises:
            UnknownPipelineInputError: If a value is given for an input the
                pipeline does not declare.
            MissingPipelineInputError: If a required input is omitted.
        """

        declared = {
            pipeline_input.name: pipeline_input
            for pipeline_input in assembled_pipeline.inputs
        }

        unknown = [name for name in pipeline_inputs if name not in declared]
        if unknown:
            raise UnknownPipelineInputError(assembled_pipeline.name, unknown)

        missing = [
            pipeline_input.name
            for pipeline_input in assembled_pipeline.inputs
            if pipeline_input.required and pipeline_input.name not in pipeline_inputs
        ]
        if missing:
            raise MissingPipelineInputError(assembled_pipeline.name, missing)

        return {
            name: pipeline_inputs.get(name, pipeline_input.default)
            for name, pipeline_input in declared.items()
        }

    def _materialize_pipeline_inputs(
        self,
        assembled_pipeline: AssembledPipeline,
        pipeline_run_id: uuid.UUID,
        resolved_inputs: Dict[str, Any],
    ) -> Dict[str, str]:
        """Materializes every resolved pipeline input exactly once, before
        any task runs.

        Args:
            assembled_pipeline: The pipeline being run.
            pipeline_run_id: This run's id.
            resolved_inputs: This run's resolved pipeline input values.

        Returns:
            Mapping from each pipeline input's name to the artifact key its
            value was materialized under.
        """

        return {
            name: self._materialize_pipeline_input(
                assembled_pipeline.name,
                pipeline_run_id,
                name,
                value,
                assembled_pipeline.input_materializers[name],
                assembled_pipeline.default_materializer,
            )
            for name, value in resolved_inputs.items()
        }

    def _materialize_pipeline_input(
        self,
        pipeline_name: str,
        pipeline_run_id: uuid.UUID,
        input_name: str,
        value: Any,
        materializer: "BaseMaterializer",
        default_materializer_cls: Type["BaseMaterializer"],
    ) -> str:
        """Materializes one pipeline input value.

        Args:
            pipeline_name: The name of the pipeline being run.
            pipeline_run_id: This run's id.
            input_name: The pipeline input's name.
            value: The pipeline input's resolved value.
            materializer: The materializer resolved for this input by the
                `Assembler`, from the pipeline function's own type hints.
            default_materializer_cls: The pipeline's own
                `default_materializer`, used by `_reconcile_materializer`
                if `materializer` turns out not to actually support `value`.

        Returns:
            The artifact key `value` was materialized under.
        """

        materializer = self._reconcile_materializer(
            materializer, value, default_materializer_cls, input_name
        )
        key = self.artifact_client.key(
            pipeline_name,
            pipeline_run_id,
            PIPELINE_INPUT_TASK_NAME,
            input_name,
        )
        self.artifact_client.save(materializer, value, key)

        return key

    def _run_task(
        self,
        assembled_pipeline: AssembledPipeline,
        assembled_task: AssembledTask,
        pipeline_input_keys: Dict[str, str],
        task_output_keys: Dict[_TaskOutputKey, str],
        pipeline_run_id: uuid.UUID,
        code_bundle_key: Optional[str],
    ) -> None:
        """Resolves `assembled_task`'s input/output artifact keys and
        dispatches it to its own `compute_backend` to actually run.

        Also records this task run's bookkeeping into the metadata store:
        `RUNNING` before anything below runs, then `SUCCEEDED`/`FAILED`
        (re-raising the original exception either way) along with whatever
        was captured on stdout/stderr (plus a traceback on failure,
        wherever the task actually ran) as `logs`, plus each output's
        artifact key once the backend confirms they were all materialized.

        Args:
            assembled_pipeline: The pipeline `assembled_task` belongs to.
            assembled_task: The task to run.
            pipeline_input_keys: Mapping from pipeline input name to the
                artifact key it was materialized under.
            task_output_keys: Mapping from `(task_name, output_name)` to
                the artifact key that output was materialized under.
                Mutated in place: each of `assembled_task`'s own output(s)
                is added to it once computed, for any downstream task to
                consume.
            pipeline_run_id: This run's id.
            code_bundle_key: This run's code bundle key, or `None` if none
                was uploaded (see `_maybe_upload_code_bundle`).
        """

        self.metadata_client.start_task_run(pipeline_run_id, assembled_task.name)

        try:
            bindings_by_input = {
                binding.target.input_name: binding
                for binding in assembled_pipeline.bindings_for(assembled_task.name)
            }
            input_keys = {
                input_name: self._resolve_input_key(
                    binding, pipeline_input_keys, task_output_keys
                )
                for input_name, binding in bindings_by_input.items()
            }
            output_keys = {
                output_name: self.artifact_client.key(
                    assembled_pipeline.name,
                    pipeline_run_id,
                    assembled_task.name,
                    output_name,
                )
                for output_name in assembled_task.output_names
            }

            logs = assembled_task.compute_backend.run(
                assembled_task,
                self.artifact_client,
                code_bundle_key,
                input_keys,
                output_keys,
                assembled_pipeline.name,
                pipeline_run_id,
                assembled_pipeline.default_materializer,
            )

            for output_name, key in output_keys.items():
                self.metadata_client.record_task_output(
                    pipeline_run_id, assembled_task.name, output_name, key
                )
                task_output_keys[(assembled_task.name, output_name)] = key
        except Exception as exc:
            self.metadata_client.finish_task_run(
                pipeline_run_id,
                assembled_task.name,
                RunStatus.FAILED,
                logs=get_captured_logs(exc),
            )
            raise

        self.metadata_client.finish_task_run(
            pipeline_run_id, assembled_task.name, RunStatus.SUCCEEDED, logs=logs
        )

    def _resolve_input_key(
        self,
        binding: IOBinding,
        pipeline_input_keys: Dict[str, str],
        task_output_keys: Dict[_TaskOutputKey, str],
    ) -> str:
        """Resolves the artifact key a bound task input should be loaded
        from.

        Args:
            binding: The `IOBinding` feeding this input.
            pipeline_input_keys: Mapping from pipeline input name to the
                artifact key it was materialized under.
            task_output_keys: Mapping from `(task_name, output_name)` to
                the artifact key that output was materialized under.

        Returns:
            The resolved artifact key.
        """

        if isinstance(binding.source, TaskOutput):
            return task_output_keys[
                (binding.source.task_name, binding.source.output_name)
            ]

        return pipeline_input_keys[binding.source.name]

    def _reconcile_materializer(
        self,
        materializer: "BaseMaterializer",
        value: Any,
        default_materializer_cls: Type["BaseMaterializer"],
        described_by: str,
    ) -> "BaseMaterializer":
        """Reconciles an assembly-time-resolved materializer against the
        actual value about to be saved - used for a pipeline input's own
        materialization, which (unlike a task's) isn't routed through
        `execute_task`.

        Python's type hints aren't enforced at runtime, so the materializer
        the `Assembler` resolved from a declared type hint can turn out not
        to actually support what a task/pipeline input produced. Rather
        than let a mismatched materializer's `_save()` fail with a
        confusing, several-layers-removed error, this checks
        `materializer.supports(value)` first and, if it fails, re-resolves
        from the value itself - the same fallback `resolve_materializer_for_
        type` uses when nothing in the registry matches a type hint.

        Args:
            materializer: The materializer resolved at assembly time.
            value: The actual value about to be saved.
            default_materializer_cls: The pipeline's own
                `default_materializer`, used if nothing in the registry
                supports `value` either.
            described_by: A human-readable description of what's being
                materialized (a pipeline input's name), used in the warning
                message.

        Returns:
            `materializer` unchanged if it already supports `value`;
            otherwise the materializer `resolve_materializer_for_value`
            resolves for it instead.
        """

        if materializer.supports(value):
            return materializer

        warnings.warn(
            f"{described_by}'s declared type doesn't match what it "
            f"actually produced (a {type(value).__name__}) - "
            f"{type(materializer).__name__} can't serialize it. Re-resolving "
            "a materializer from the actual value instead. Consider fixing "
            "the type hint.",
            MaterializerMismatchWarning,
            stacklevel=2,
        )

        return resolve_materializer_for_value(value, default_materializer_cls)

    def _resolve_output_value(
        self,
        assembled_pipeline: AssembledPipeline,
        pipeline_input_keys: Dict[str, str],
        task_output_keys: Dict[_TaskOutputKey, str],
    ) -> Any:
        """Resolves the pipeline's single output value.

        Args:
            assembled_pipeline: The pipeline being run.
            pipeline_input_keys: Mapping from pipeline input name to the
                artifact key it was materialized under.
            task_output_keys: Mapping from `(task_name, output_name)` to
                the artifact key that output was materialized under.

        Returns:
            The pipeline's output value, loaded from whichever of a task
            output or a pipeline input it is a reference to.
        """

        source = assembled_pipeline.output.source

        if isinstance(source, TaskOutput):
            key = task_output_keys[(source.task_name, source.output_name)]
        else:
            key = pipeline_input_keys[source.name]

        return self._load(key)

    def _load(self, key: str) -> Any:
        """Loads an already-materialized artifact.

        Resolves the materializer from the artifact's own stored metadata
        rather than from any consuming task's declared type: a pipeline
        input has no such thing to begin with, and reusing whichever
        materializer actually produced the bytes avoids relying on a
        consumer's independently-resolved one just happening to match.

        Args:
            key: The artifact's key.

        Returns:
            The deserialized value.
        """

        materializer = resolve_materializer_from_artifact(self.artifact_client, key)

        return self.artifact_client.load(materializer, key)


def run_locally(
    assembled_pipeline: AssembledPipeline,
    artifact_client: Optional[ArtifactClient] = None,
    metadata_client: Optional[MetadataClient] = None,
    **pipeline_inputs: Any,
) -> Any:
    """Convenience wrapper around
    `LocalRunner(artifact_client, metadata_client).run(...)`.

    Args:
        assembled_pipeline: The pipeline to run.
        artifact_client: The client to materialize pipeline inputs and task
            outputs through. Defaults to a fresh `ArtifactClient()` if
            omitted.
        metadata_client: The client to record this run's bookkeeping into.
            Defaults to a fresh `MetadataClient()` if omitted.
        **pipeline_inputs: Values for the pipeline's declared inputs. Any
            input with a default may be omitted.

    Returns:
        The pipeline's single output value, or `None` if it declares none.
    """

    return LocalRunner(artifact_client, metadata_client).run(
        assembled_pipeline, **pipeline_inputs
    )
