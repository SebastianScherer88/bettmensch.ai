"""`LocalRunner`: executes an `AssembledPipeline` in-process."""

from __future__ import annotations

import contextlib
import io
import traceback
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type

from ..artifact_store import BaseArtifactStore, LocalArtifactStore
from ..assembler.recording import record_assembly
from ..io_binding import IOBinding, TaskOutput
from ..materializers import (
    resolve_materializer_for_value,
    resolve_materializer_from_artifact,
)
from ..metadata_store import BaseMetadataStore, LocalMetadataStore, RunStatus
from ..pipeline.assembled_pipeline import AssembledPipeline
from ..task.assembled_task import AssembledTask
from .exceptions import (
    MaterializerMismatchWarning,
    MissingPipelineInputError,
    UnknownPipelineInputError,
)

if TYPE_CHECKING:
    from ..materializers.base_materializer import BaseMaterializer

_TaskOutputKey = Tuple[str, str]

# Reserved "task" sentinel for a pipeline input's own materialized artifact
# key - can never collide with a real (dasherized) task name, the same way
# `code_bundler.CODE_BUNDLE_TASK_NAME` can't.
PIPELINE_INPUT_TASK_NAME = "__pipeline_input__"


class LocalRunner:
    """Executes an `AssembledPipeline` locally.

    Runs each `AssembledTask`'s underlying function, rank by rank in the
    pipeline's topological order. Only two kinds of value are ever
    materialized (saved) to the artifact store: each of the pipeline's own
    inputs, exactly once, before any task runs; and each of a task's
    output(s), once computed - a multi-output (`NamedTuple`/`TypedDict`
    returning) task's fields/keys are each materialized independently,
    under their own key, rather than as one combined blob. A task input
    bound to a `TaskOutput` or
    `PipelineInput` is *loaded* from whichever of those an `IOBinding`
    points it at - never independently re-saved - and a static/literal
    input is used directly, in memory, never touching the store at all.
    Tasks within a rank are independent of each other (per
    `AssembledPipeline.task_ranks`) but are still run sequentially here;
    `LocalRunner` does not itself parallelize them.

    Deliberately independent of the `Assembler`: running an already-valid
    `AssembledPipeline` is a separate concern from assembling and
    validating one. Returns the pipeline's single output value directly (or
    `None` if it declares none) - a pipeline, like a `Task`, produces
    exactly one output.

    Also records this run's bookkeeping into a `BaseMetadataStore`: first,
    `assembled_pipeline`'s current structure (via `assembler.record_assembly`
    - a no-op if it's unchanged since the last recorded assembly of this
    pipeline), then the pipeline run's own status (`RUNNING` once inputs
    validate, then `SUCCEEDED`/`FAILED`), each task run's status, and each
    task output's artifact key. This is a distinct concern from
    materialization - the artifact store persists *values*, the metadata
    store persists *what happened* - and `LocalRunner` is only one of
    (eventually) several orchestrators expected to write into the same
    metadata store, local or remote, hence keeping the two separate.
    """

    def __init__(
        self,
        artifact_store: Optional[BaseArtifactStore] = None,
        metadata_store: Optional[BaseMetadataStore] = None,
    ):
        """Initializes the runner.

        Args:
            artifact_store: The store to materialize pipeline inputs and
                task outputs through. Defaults to a fresh
                `LocalArtifactStore()` if omitted.
            metadata_store: The store to record this run's bookkeeping
                (status, timing, task output keys) into. Defaults to a
                fresh `LocalMetadataStore()` if omitted.
        """

        self.artifact_store = artifact_store or LocalArtifactStore()
        self.metadata_store = metadata_store or LocalMetadataStore()

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

        pipeline_assembly_id = record_assembly(self.metadata_store, assembled_pipeline)

        pipeline_run_id = uuid.uuid4()
        self.metadata_store.start_pipeline_run(
            assembled_pipeline.name, pipeline_run_id, pipeline_assembly_id
        )

        try:
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
                    )

            result = (
                None
                if assembled_pipeline.output is None
                else self._resolve_output_value(
                    assembled_pipeline, pipeline_input_keys, task_output_keys
                )
            )
        except Exception:
            self.metadata_store.finish_pipeline_run(pipeline_run_id, RunStatus.FAILED)
            raise

        self.metadata_store.finish_pipeline_run(pipeline_run_id, RunStatus.SUCCEEDED)

        return result

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
        key = self.artifact_store.key(
            pipeline_name,
            pipeline_run_id,
            PIPELINE_INPUT_TASK_NAME,
            input_name,
        )
        self.artifact_store.save(materializer, value, key)

        return key

    def _run_task(
        self,
        assembled_pipeline: AssembledPipeline,
        assembled_task: AssembledTask,
        pipeline_input_keys: Dict[str, str],
        task_output_keys: Dict[_TaskOutputKey, str],
        pipeline_run_id: uuid.UUID,
    ) -> None:
        """Resolves `assembled_task`'s inputs, calls its function, and
        materializes each of its output(s).

        Also records this task run's bookkeeping into the metadata store:
        `RUNNING` before anything below runs, then `SUCCEEDED`/`FAILED`
        (re-raising the original exception either way) along with whatever
        the task function wrote to stdout/stderr (plus a traceback on
        failure) as `logs`, plus each output's artifact key as it's
        materialized.

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
        """

        self.metadata_store.start_task_run(pipeline_run_id, assembled_task.name)
        log_buffer = io.StringIO()

        try:
            bindings_by_input = {
                binding.target.input_name: binding
                for binding in assembled_pipeline.bindings_for(assembled_task.name)
            }

            kwargs = {
                input_name: self._resolve_input(
                    assembled_task,
                    input_name,
                    bindings_by_input.get(input_name),
                    pipeline_input_keys,
                    task_output_keys,
                )
                for input_name in assembled_task.task.signature.parameters
            }

            with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(
                log_buffer
            ):
                value = assembled_task.func(**kwargs)

            for output_name in assembled_task.output_names:
                output_value = self._extract_output_value(
                    assembled_task, value, output_name
                )
                materializer = self._reconcile_materializer(
                    assembled_task.materializers[output_name],
                    output_value,
                    assembled_pipeline.default_materializer,
                    f"{assembled_task.name}.{output_name}",
                )
                key = self.artifact_store.key(
                    assembled_pipeline.name,
                    pipeline_run_id,
                    assembled_task.name,
                    output_name,
                )
                self.artifact_store.save(materializer, output_value, key)
                self.metadata_store.record_task_output(
                    pipeline_run_id, assembled_task.name, output_name, key
                )
                task_output_keys[(assembled_task.name, output_name)] = key
        except Exception:
            log_buffer.write(traceback.format_exc())
            self.metadata_store.finish_task_run(
                pipeline_run_id,
                assembled_task.name,
                RunStatus.FAILED,
                logs=log_buffer.getvalue() or None,
            )
            raise

        self.metadata_store.finish_task_run(
            pipeline_run_id,
            assembled_task.name,
            RunStatus.SUCCEEDED,
            logs=log_buffer.getvalue() or None,
        )

    def _reconcile_materializer(
        self,
        materializer: "BaseMaterializer",
        value: Any,
        default_materializer_cls: Type["BaseMaterializer"],
        described_by: str,
    ) -> "BaseMaterializer":
        """Reconciles an assembly-time-resolved materializer against the
        actual value about to be saved.

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
                materialized (a task output or pipeline input's name), used
                in the warning message.

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

    def _extract_output_value(
        self, assembled_task: AssembledTask, value: Any, output_name: str
    ) -> Any:
        """Extracts one named output's value from a task function's return
        value.

        Args:
            assembled_task: The task that was just run.
            value: The value `assembled_task.func` returned.
            output_name: The output name to extract.

        Returns:
            `value` itself for an ordinary (single-output) task; one field
            (`NamedTuple`) or key (`TypedDict`) of it for a multi-output
            one.
        """

        task = assembled_task.task

        if task.is_typed_dict_output:
            return value[output_name]

        if task.is_named_tuple_output:
            return getattr(value, output_name)

        return value

    def _resolve_input(
        self,
        assembled_task: AssembledTask,
        input_name: str,
        binding: Optional[IOBinding],
        pipeline_input_keys: Dict[str, str],
        task_output_keys: Dict[_TaskOutputKey, str],
    ) -> Any:
        """Resolves the value for one of `assembled_task`'s inputs.

        No task input is ever independently materialized: a static/literal
        input is used directly, in memory; a `TaskOutput`- or
        `PipelineInput`-bound input is loaded from whichever of those was
        already materialized (a task's output when it ran, or a pipeline
        input once, up front) - never re-saved, however many tasks consume
        it.

        Args:
            assembled_task: The task `input_name` belongs to.
            input_name: The name of the input to resolve.
            binding: The `IOBinding` feeding `input_name`, if any (`None`
                for a static/literal input).
            pipeline_input_keys: Mapping from pipeline input name to the
                artifact key it was materialized under.
            task_output_keys: Mapping from `(task_name, output_name)` to
                the artifact key that output was materialized under.

        Returns:
            The resolved, deserialized input value.
        """

        if binding is None:
            return assembled_task.static_inputs[input_name]

        if isinstance(binding.source, TaskOutput):
            key = task_output_keys[
                (binding.source.task_name, binding.source.output_name)
            ]
        else:
            key = pipeline_input_keys[binding.source.name]

        return self._load(key)

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

        materializer = resolve_materializer_from_artifact(self.artifact_store, key)

        return self.artifact_store.load(materializer, key)


def run_locally(
    assembled_pipeline: AssembledPipeline,
    artifact_store: Optional[BaseArtifactStore] = None,
    metadata_store: Optional[BaseMetadataStore] = None,
    **pipeline_inputs: Any,
) -> Any:
    """Convenience wrapper around
    `LocalRunner(artifact_store, metadata_store).run(...)`.

    Args:
        assembled_pipeline: The pipeline to run.
        artifact_store: The store to materialize pipeline inputs and task
            outputs through. Defaults to a fresh `LocalArtifactStore()` if
            omitted.
        metadata_store: The store to record this run's bookkeeping into.
            Defaults to a fresh `LocalMetadataStore()` if omitted.
        **pipeline_inputs: Values for the pipeline's declared inputs. Any
            input with a default may be omitted.

    Returns:
        The pipeline's single output value, or `None` if it declares none.
    """

    return LocalRunner(artifact_store, metadata_store).run(
        assembled_pipeline, **pipeline_inputs
    )
