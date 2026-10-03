"""`RemoteRunner`: invokes an already-`RegisteredPipeline`'s own remote
execution (e.g. a Step Functions state machine running the whole DAG) - the
sole entry point for that. `LocalRunner` may dispatch an individual *task*
to remote compute, but it never dispatches a whole *pipeline's* execution to
a remote orchestrator; that is this class's job.
"""

from __future__ import annotations

import json
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union

import boto3

from ..artifact_store import BaseArtifactStore, LocalArtifactStore
from ..assembler.recording import record_assembly_from_dicts
from ..code_bundler import CodeBundler
from ..materializers import (
    MATERIALIZER_BY_NAME,
    DefaultMaterializer,
    resolve_materializer_for_value,
    resolve_materializer_from_artifact,
)
from ..metadata_store import BaseMetadataStore, LocalMetadataStore, RunStatus
from .exceptions import (
    MaterializerMismatchWarning,
    MissingPipelineInputError,
    RemoteTaskExecutionError,
    UnknownPipelineInputError,
    UnsupportedBackendError,
)
from .registered_pipeline import RegisteredPipeline

_TaskOutputKey = Tuple[str, str]

# Mirrors `local_runner.PIPELINE_INPUT_TASK_NAME` exactly - the reserved
# "task name" a pipeline input's own materialized artifact is keyed under.
# Also what `compilers.stepfunctions.asl` builds artifact-key expressions
# against for a task input bound to a pipeline input.
PIPELINE_INPUT_TASK_NAME = "__pipeline_input__"

_SUPPORTED_BACKENDS = frozenset({"aws_stepfunctions"})

# Step Functions execution history event types that mark a `Task` state's
# own success/failure - as opposed to `Parallel`/`Map` container states,
# which this runner doesn't need to track individually since every task
# within one still gets its own `TaskState*`/`TaskFailed` events.
_TASK_FAILURE_EVENT_TYPES = frozenset({"TaskFailed", "TaskTimedOut", "TaskAborted"})


class RemoteRunner:
    """Runs a `RegisteredPipeline` to completion on its own registered
    backend orchestrator (only AWS Step Functions today).

    Needs no live `AssembledPipeline`/`CompiledPipeline` - a
    `RegisteredPipeline` is a lightweight enough reference (a state machine
    ARN, the DAG structure/inputs recorded at registration time) that
    running it needs nothing more of the pipeline's own Python code beyond
    a fresh code bundle (every task a registered pipeline runs is on remote
    compute, so - unlike `LocalRunner` - one is always uploaded).

    Mirrors bookkeeping into the same `BaseMetadataStore` `LocalRunner`
    writes into, per task, as the remote execution progresses: an assembly
    record for `registered_pipeline`'s own stored structure (deduped
    exactly like `LocalRunner`'s), then the run's own status, then each
    task's status and output artifact key(s) as Step Functions' execution
    history reports them finishing. This is what lets a `RemoteRunner`-
    driven run render identically to a `LocalRunner`-driven one in a Runs
    UI - same assembly record, same per-task drill-down - despite no
    Python task code having run in this process at all.
    """

    def __init__(
        self,
        artifact_store: Optional[BaseArtifactStore] = None,
        metadata_store: Optional[BaseMetadataStore] = None,
        project_root: Optional[Union[str, Path]] = None,
        poll_interval_seconds: float = 5.0,
    ):
        """Initializes the runner.

        Args:
            artifact_store: The store `registered_pipeline`'s tasks read
                their inputs from and write their outputs to - must be
                reachable from wherever the backend actually runs (e.g. an
                `S3ArtifactStore`; a `LocalArtifactStore` only makes sense
                against a mocked backend, e.g. in tests). Defaults to a
                fresh `LocalArtifactStore()` if omitted, matching
                `LocalRunner`'s own default.
            metadata_store: The store this run's bookkeeping is recorded
                into - normally the same one `registered_pipeline` was
                looked up from. Defaults to a fresh `LocalMetadataStore()`
                if omitted.
            project_root: The directory `CodeBundler` bundles for this run.
                Defaults to the current working directory if omitted.
            poll_interval_seconds: How long to wait between polls of the
                backend execution's status while it's still running.
        """

        self.artifact_store = artifact_store or LocalArtifactStore()
        self.metadata_store = metadata_store or LocalMetadataStore()
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        self.poll_interval_seconds = poll_interval_seconds

    def run(self, registered_pipeline: RegisteredPipeline, **pipeline_inputs: Any) -> Any:
        """Runs `registered_pipeline` to completion on its registered
        backend.

        Args:
            registered_pipeline: The pipeline to run.
            **pipeline_inputs: Values for the pipeline's declared inputs.
                Any input with a default may be omitted.

        Returns:
            The pipeline's single output value, or `None` if it declares
            none.

        Raises:
            UnsupportedBackendError: If `registered_pipeline.backend` isn't
                one this runner knows how to invoke.
            MissingPipelineInputError: If a required input is omitted.
            UnknownPipelineInputError: If a value is given for an input the
                pipeline does not declare.
            RemoteTaskExecutionError: If the backend execution doesn't
                finish `SUCCEEDED`.
        """

        if registered_pipeline.backend not in _SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(
                f"RemoteRunner cannot invoke pipeline "
                f"{registered_pipeline.pipeline_name!r}'s registration "
                f"{registered_pipeline.pipeline_registration_id} - backend "
                f"{registered_pipeline.backend!r} isn't supported (only "
                f"{sorted(_SUPPORTED_BACKENDS)} are)."
            )

        resolved_inputs = self._resolve_pipeline_inputs(registered_pipeline, pipeline_inputs)

        pipeline_assembly_id = record_assembly_from_dicts(
            self.metadata_store,
            registered_pipeline.pipeline_name,
            registered_pipeline.dag_structure,
            registered_pipeline.pipeline_inputs,
        )

        pipeline_run_id = uuid.uuid4()
        self.metadata_store.start_pipeline_run(
            registered_pipeline.pipeline_name, pipeline_run_id, pipeline_assembly_id
        )

        try:
            code_bundle_key = CodeBundler(self.project_root).bundle_and_upload(
                self.artifact_store, registered_pipeline.pipeline_name, pipeline_run_id
            )
            pipeline_input_keys = self._materialize_pipeline_inputs(
                registered_pipeline, pipeline_run_id, resolved_inputs
            )

            execution_arn = self._start_execution(
                registered_pipeline, pipeline_run_id, code_bundle_key
            )
            task_output_keys = self._track_execution(
                registered_pipeline, pipeline_run_id, execution_arn
            )

            output = registered_pipeline.dag_structure.get("output")
            result = (
                None
                if output is None
                else self._resolve_output_value(output, pipeline_input_keys, task_output_keys)
            )
        except Exception:
            self.metadata_store.finish_pipeline_run(pipeline_run_id, RunStatus.FAILED)
            raise

        self.metadata_store.finish_pipeline_run(pipeline_run_id, RunStatus.SUCCEEDED)

        return result

    def _resolve_pipeline_inputs(
        self, registered_pipeline: RegisteredPipeline, pipeline_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validates and resolves the values a `run()` call was given
        against `registered_pipeline.pipeline_inputs`.

        Args:
            registered_pipeline: The pipeline being run.
            pipeline_inputs: The raw values `run()` was called with.

        Returns:
            A mapping from every declared input's name to its resolved
            value (the given one, or its recorded default if omitted).

        Raises:
            UnknownPipelineInputError: If a value is given for an input the
                pipeline does not declare.
            MissingPipelineInputError: If a required input is omitted.
        """

        declared = registered_pipeline.pipeline_inputs

        unknown = [name for name in pipeline_inputs if name not in declared]
        if unknown:
            raise UnknownPipelineInputError(registered_pipeline.pipeline_name, unknown)

        missing = [
            name
            for name, spec in declared.items()
            if spec["required"] and name not in pipeline_inputs
        ]
        if missing:
            raise MissingPipelineInputError(registered_pipeline.pipeline_name, missing)

        return {
            name: pipeline_inputs.get(name, spec["default"]) for name, spec in declared.items()
        }

    def _materialize_pipeline_inputs(
        self,
        registered_pipeline: RegisteredPipeline,
        pipeline_run_id: uuid.UUID,
        resolved_inputs: Dict[str, Any],
    ) -> Dict[str, str]:
        """Materializes every resolved pipeline input exactly once, before
        the backend execution starts - mirrors `LocalRunner`'s own pipeline
        input materialization, under the identical deterministic key
        scheme, so the state machine's `$.pipeline_name`/`$.pipeline_run_id`
        based key expressions resolve to the right place.

        Args:
            registered_pipeline: The pipeline being run.
            pipeline_run_id: This run's id.
            resolved_inputs: This run's resolved pipeline input values.

        Returns:
            Mapping from each pipeline input's name to the artifact key its
            value was materialized under.
        """

        return {
            name: self._materialize_pipeline_input(
                registered_pipeline.pipeline_name,
                pipeline_run_id,
                name,
                value,
                registered_pipeline.pipeline_inputs[name]["materializer"],
            )
            for name, value in resolved_inputs.items()
        }

    def _materialize_pipeline_input(
        self,
        pipeline_name: str,
        pipeline_run_id: uuid.UUID,
        input_name: str,
        value: Any,
        materializer_name: str,
    ) -> str:
        """Materializes one pipeline input value.

        Args:
            pipeline_name: The name of the pipeline being run.
            pipeline_run_id: This run's id.
            input_name: The pipeline input's name.
            value: The pipeline input's resolved value.
            materializer_name: The stable name of the materializer recorded
                for this input at registration time.

        Returns:
            The artifact key `value` was materialized under.
        """

        materializer_cls = MATERIALIZER_BY_NAME.get(materializer_name, DefaultMaterializer)
        materializer = materializer_cls()

        if not materializer.supports(value):
            warnings.warn(
                f"{input_name!r}'s declared type doesn't match what it was "
                f"actually given (a {type(value).__name__}) - "
                f"{materializer_cls.__name__} can't serialize it. "
                "Re-resolving a materializer from the actual value instead.",
                MaterializerMismatchWarning,
                stacklevel=2,
            )
            materializer = resolve_materializer_for_value(value, DefaultMaterializer)

        key = self.artifact_store.key(
            pipeline_name, pipeline_run_id, PIPELINE_INPUT_TASK_NAME, input_name
        )
        self.artifact_store.save(materializer, value, key)

        return key

    def _start_execution(
        self,
        registered_pipeline: RegisteredPipeline,
        pipeline_run_id: uuid.UUID,
        code_bundle_key: str,
    ) -> str:
        """Starts the registered state machine's own execution.

        Args:
            registered_pipeline: The pipeline being run.
            pipeline_run_id: This run's id.
            code_bundle_key: This run's freshly-uploaded code bundle key.

        Returns:
            The new execution's ARN.
        """

        execution_input = {
            "pipeline_name": registered_pipeline.pipeline_name,
            "pipeline_run_id": str(pipeline_run_id),
            "code_bundle_key": code_bundle_key,
        }

        response = boto3.client("stepfunctions").start_execution(
            stateMachineArn=registered_pipeline.backend_metadata["state_machine_arn"],
            name=f"{registered_pipeline.pipeline_name}-{pipeline_run_id}",
            input=json.dumps(execution_input),
        )

        return response["executionArn"]

    def _track_execution(
        self,
        registered_pipeline: RegisteredPipeline,
        pipeline_run_id: uuid.UUID,
        execution_arn: str,
    ) -> Dict[_TaskOutputKey, str]:
        """Polls the execution to completion, mirroring each task's own
        progress into the metadata store as Step Functions reports it.

        A `TaskStateEntered` event for a state matching one of this
        pipeline's task names starts that task's bookkeeping;
        `TaskStateExited` finishes it `SUCCEEDED` and records each of its
        declared outputs under their deterministic artifact key (Step
        Functions' own `.sync` service integration guarantees the task
        actually finished writing them by the time its state exits). A
        `TaskFailed`/`TaskTimedOut`/`TaskAborted` event has no `name` field
        of its own, so the failing task is identified by walking back
        through `previousEventId` to the `TaskStateEntered` event that
        opened it - the same approach the Step Functions console itself
        uses to attribute a failure to a state.

        Args:
            registered_pipeline: The pipeline being run.
            pipeline_run_id: This run's id.
            execution_arn: The execution to track.

        Returns:
            Mapping from `(task_name, output_name)` to the artifact key
            that output was recorded under.

        Raises:
            RemoteTaskExecutionError: If the execution doesn't finish
                `SUCCEEDED`.
        """

        client = boto3.client("stepfunctions")
        task_names = {task["name"] for task in registered_pipeline.dag_structure["tasks"]}
        outputs_by_task = {
            task["name"]: task["outputs"] for task in registered_pipeline.dag_structure["tasks"]
        }

        task_output_keys: Dict[_TaskOutputKey, str] = {}
        started_tasks: Set[str] = set()
        finished_tasks: Set[str] = set()

        while True:
            status = client.describe_execution(executionArn=execution_arn)["status"]
            events = client.get_execution_history(executionArn=execution_arn)["events"]
            events_by_id = {event["id"]: event for event in events}

            for event in events:
                event_type = event["type"]

                if event_type == "TaskStateEntered":
                    name = event["stateEnteredEventDetails"]["name"]
                    if name in task_names and name not in started_tasks:
                        self.metadata_store.start_task_run(pipeline_run_id, name)
                        started_tasks.add(name)

                elif event_type == "TaskStateExited":
                    name = event["stateExitedEventDetails"]["name"]
                    if name in task_names and name not in finished_tasks:
                        for output_name in outputs_by_task[name]:
                            key = self.artifact_store.key(
                                registered_pipeline.pipeline_name,
                                pipeline_run_id,
                                name,
                                output_name,
                            )
                            self.metadata_store.record_task_output(
                                pipeline_run_id, name, output_name, key
                            )
                            task_output_keys[(name, output_name)] = key
                        self.metadata_store.finish_task_run(
                            pipeline_run_id, name, RunStatus.SUCCEEDED
                        )
                        finished_tasks.add(name)

                elif event_type in _TASK_FAILURE_EVENT_TYPES:
                    name = self._find_enclosing_task_name(events_by_id, event)
                    if name in task_names and name not in finished_tasks:
                        self.metadata_store.finish_task_run(
                            pipeline_run_id, name, RunStatus.FAILED
                        )
                        finished_tasks.add(name)

            if status != "RUNNING":
                break

            time.sleep(self.poll_interval_seconds)

        if status != "SUCCEEDED":
            raise RemoteTaskExecutionError(
                f"Pipeline {registered_pipeline.pipeline_name!r}'s remote "
                f"execution {execution_arn} ended with status {status!r}."
            )

        return task_output_keys

    def _find_enclosing_task_name(
        self, events_by_id: Dict[int, Dict[str, Any]], event: Dict[str, Any]
    ) -> Optional[str]:
        """Walks `event`'s `previousEventId` chain back to the
        `TaskStateEntered` event that opened the task it happened within.

        Args:
            events_by_id: Every execution history event seen so far, keyed
                by its own `id`.
            event: The event to trace back from (typically a
                `TaskFailed`/`TaskTimedOut`/`TaskAborted` event, which
                carries no `name` of its own).

        Returns:
            The enclosing task's state name, or `None` if it couldn't be
            traced (e.g. a failure at the top-level execution, outside any
            task state).
        """

        current = event
        seen_ids: Set[int] = set()

        while current is not None:
            event_id = current.get("id")
            if event_id in seen_ids:
                return None
            seen_ids.add(event_id)

            if current["type"] == "TaskStateEntered":
                return current["stateEnteredEventDetails"]["name"]

            current = events_by_id.get(current.get("previousEventId"))

        return None

    def _resolve_output_value(
        self,
        output: Dict[str, Any],
        pipeline_input_keys: Dict[str, str],
        task_output_keys: Dict[_TaskOutputKey, str],
    ) -> Any:
        """Resolves the pipeline's single output value from its serialized
        `dag_structure["output"]` source reference.

        Args:
            output: `registered_pipeline.dag_structure["output"]`
                (`{"name", "source": {"kind", ...}}`).
            pipeline_input_keys: Mapping from pipeline input name to the
                artifact key it was materialized under.
            task_output_keys: Mapping from `(task_name, output_name)` to
                the artifact key that output was recorded under.

        Returns:
            The pipeline's output value, loaded from whichever of a task
            output or a pipeline input it is a reference to.
        """

        source = output["source"]

        if source["kind"] == "task_output":
            key = task_output_keys[(source["task"], source["output"])]
        else:
            key = pipeline_input_keys[source["name"]]

        return self._load(key)

    def _load(self, key: str) -> Any:
        """Loads an already-materialized artifact, resolving its
        materializer from its own stored metadata sidecar.

        Args:
            key: The artifact's key.

        Returns:
            The deserialized value.
        """

        materializer = resolve_materializer_from_artifact(self.artifact_store, key)

        return self.artifact_store.load(materializer, key)
