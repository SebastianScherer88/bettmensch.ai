"""Pure Amazon States Language (ASL) construction from an `AssembledPipeline`.

Best-effort: this project has no real AWS account to validate the exact
`States.Format` intrinsic-function expressions below against a live Step
Functions execution. The *structural* mapping (a `Parallel` state for a
multi-task rank, a `Task` state per task, sequential chaining via `Next`)
is straightforward and well-tested; the key-construction expressions are a
solid starting point, not a verified guarantee - the state machine only
ever carries artifact *keys* through its `Parameters`, matching how
`execute_task` already resolves inputs/outputs by key, so each key is
built from the execution's own `pipeline_name`/`pipeline_run_id` input
fields (the only two things not known until an execution actually starts)
plus the task/output names, which are fixed at compile time.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Type

from ...compute.aws_batch_compute_backend import AwsBatchComputeBackend
from ...compute.aws_lambda_compute_backend import AwsLambdaComputeBackend
from ...io_binding import TaskOutput
from ...pipeline.assembled_pipeline import AssembledPipeline
from ...task.assembled_task import AssembledTask
from ..exceptions import UnsupportedComputeBackendError

# The one, reserved "task name" a pipeline input's own artifact key is
# stored under - matches `runner.local_runner.PIPELINE_INPUT_TASK_NAME`.
_PIPELINE_INPUT_TASK_NAME = "__pipeline_input__"


def _key_expression(task_name: str, artifact_name: str) -> str:
    """A `States.Format` expression for one artifact's key: the execution's
    own `pipeline_name`/`pipeline_run_id` plus this task/artifact name pair
    (both fixed at compile time), joined the same way `BaseArtifactStore.
    key()` joins them.
    """

    return (
        f"States.Format('{{}}/{{}}/{task_name}/{artifact_name}', "
        "$.pipeline_name, $.pipeline_run_id)"
    )


def _keys_json_expression(pairs: List[tuple]) -> str:
    """A `States.Format` expression building the JSON string
    `remote_entrypoint` expects for `--input-keys`/`--output-keys`, from
    `[(input_or_output_name, (task_name, artifact_name)), ...]`.
    """

    if not pairs:
        return "'{}'"

    template = "{" + ",".join(f'\\"{name}\\":\\"{{}}\\"' for name, _ in pairs) + "}"
    args = ", ".join(_key_expression(task, artifact) for _, (task, artifact) in pairs)

    return f"States.Format('{template}', {args})"


def _input_key_pairs(assembled_pipeline: AssembledPipeline, assembled_task: AssembledTask):
    """`[(input_name, (source_task_name, source_artifact_name))]` for
    `assembled_task`'s *bound* (non-static) inputs.
    """

    pairs = []
    for binding in assembled_pipeline.bindings_for(assembled_task.name):
        if isinstance(binding.source, TaskOutput):
            pairs.append(
                (binding.target.input_name, (binding.source.task_name, binding.source.output_name))
            )
        else:
            pairs.append(
                (binding.target.input_name, (_PIPELINE_INPUT_TASK_NAME, binding.source.name))
            )
    return pairs


def _command(assembled_task: AssembledTask, default_materializer_cls: Type) -> List[str]:
    """The remote entrypoint's fixed (compile-time-known) command-line
    arguments - everything except `--input-keys`/`--output-keys`, which
    need the JSONPath expressions above.
    """

    func = assembled_task.func

    return [
        "python",
        "-m",
        "bettmensch_ai.pipelines.runner.remote_entrypoint",
        "--task-module",
        func.__module__,
        "--task-qualname",
        func.__qualname__,
        "--default-materializer-module",
        default_materializer_cls.__module__,
        "--default-materializer-qualname",
        default_materializer_cls.__qualname__,
        "--static-inputs",
        json.dumps(assembled_task.static_inputs),
    ]


def build_task_state(
    assembled_pipeline: AssembledPipeline, assembled_task: AssembledTask
) -> Dict[str, Any]:
    """Builds one ASL `Task` state for `assembled_task`.

    Args:
        assembled_pipeline: The pipeline `assembled_task` belongs to.
        assembled_task: The task to build a state for.

    Returns:
        An ASL `Task` state dict, keyed by nothing yet (the caller places
        it under its own state name).

    Raises:
        UnsupportedComputeBackendError: If `assembled_task.compute_backend`
            isn't `AwsBatchComputeBackend`/`AwsLambdaComputeBackend`.
    """

    backend = assembled_task.compute_backend
    input_keys_expr = _keys_json_expression(
        _input_key_pairs(assembled_pipeline, assembled_task)
    )
    output_keys_expr = _keys_json_expression(
        [(name, (assembled_task.name, name)) for name in assembled_task.output_names]
    )
    command = _command(assembled_task, assembled_pipeline.default_materializer)
    result_path = f"$.{assembled_task.name}"

    if isinstance(backend, AwsBatchComputeBackend):
        return {
            "Type": "Task",
            "Resource": "arn:aws:states:::batch:submitJob.sync",
            "Parameters": {
                "JobName.$": (
                    f"States.Format('{assembled_pipeline.name}-{assembled_task.name}-{{}}', "
                    "$$.Execution.Name)"
                ),
                "JobQueue": backend.config.job_queue,
                "JobDefinition": backend.config.job_definition,
                "ContainerOverrides": {
                    "Command.$": (
                        "States.Array("
                        + ", ".join(f"'{part}'" for part in command)
                        + ", '--code-bundle-key', $.code_bundle_key, "
                        f"'--input-keys', {input_keys_expr}, "
                        f"'--output-keys', {output_keys_expr})"
                    ),
                },
            },
            "ResultPath": result_path,
        }

    if isinstance(backend, AwsLambdaComputeBackend):
        return {
            "Type": "Task",
            "Resource": "arn:aws:states:::lambda:invoke",
            "Parameters": {
                "FunctionName": backend.config.function_name,
                "Payload": {
                    "task_module": assembled_task.func.__module__,
                    "task_qualname": assembled_task.func.__qualname__,
                    "default_materializer_module": assembled_pipeline.default_materializer.__module__,
                    "default_materializer_qualname": assembled_pipeline.default_materializer.__qualname__,
                    "static_inputs": assembled_task.static_inputs,
                    "code_bundle_key.$": "$.code_bundle_key",
                    "input_keys.$": input_keys_expr,
                    "output_keys.$": output_keys_expr,
                },
            },
            "ResultPath": result_path,
        }

    raise UnsupportedComputeBackendError(
        f"Task {assembled_task.name!r} is on {type(backend).__name__}, not "
        "an AWS backend - every task must be placed on aws_batch(...)/"
        "aws_lambda(...) before compiling to Step Functions."
    )


def build_state_machine_definition(assembled_pipeline: AssembledPipeline) -> Dict[str, Any]:
    """Builds the full ASL state machine definition for `assembled_pipeline`.

    A rank with more than one task becomes a `Parallel` state whose
    branches are single-task chains; a single-task rank is just that task's
    own state, in the sequential chain. Ranks are chained via `Next`; the
    last rank's state(s) get `End: true`.

    Args:
        assembled_pipeline: The pipeline to compile. Every task must be
            placed on an AWS compute backend (validated by
            `build_task_state`).

    Returns:
        `{"StartAt": ..., "States": {...}}`.

    Raises:
        UnsupportedComputeBackendError: If any task is still on
            `LocalComputeBackend`.
    """

    states: Dict[str, Any] = {}
    rank_state_names: List[str] = []

    for rank_index, rank in enumerate(assembled_pipeline.task_ranks):
        if len(rank) == 1:
            task = rank[0]
            states[task.name] = build_task_state(assembled_pipeline, task)
            rank_state_names.append(task.name)
        else:
            rank_name = f"rank-{rank_index}"
            branches = []
            for task in rank:
                task_state = build_task_state(assembled_pipeline, task)
                task_state["End"] = True
                branches.append({"StartAt": task.name, "States": {task.name: task_state}})
            states[rank_name] = {"Type": "Parallel", "Branches": branches}
            rank_state_names.append(rank_name)

    for i, name in enumerate(rank_state_names):
        if i + 1 < len(rank_state_names):
            states[name]["Next"] = rank_state_names[i + 1]
        else:
            states[name]["End"] = True

    return {
        "StartAt": rank_state_names[0] if rank_state_names else None,
        "States": states,
    }
