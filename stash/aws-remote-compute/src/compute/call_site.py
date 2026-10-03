"""`aws_batch`/`aws_lambda`: pin a task call to remote compute at the
pipeline definition's own call site - decoupling task orchestration from
the task function itself, per Metaflow's own decorator-pattern spirit,
just applied at the call site rather than the function definition.
"""

from __future__ import annotations

from typing import Any, TypeVar

from ..context import PipelineAssemblyError, get_active_context
from ..io_binding import TaskOutput
from .aws_batch_compute_backend import AwsBatchComputeBackend, AwsBatchConfig
from .aws_lambda_compute_backend import AwsLambdaComputeBackend, AwsLambdaConfig
from .base_compute_backend import BaseComputeBackend

T = TypeVar("T")


def _task_name_of(task_output_like: Any) -> str:
    """Extracts the shared task name from whatever a traced task call
    returned.

    A task call while tracing returns a single `TaskOutput` (ordinary
    task), a `NamedTuple` of them (multi-output), or a `dict` of them
    (`TypedDict`-returning) - every output of one call shares the same
    task name regardless of which shape it came back as.

    Args:
        task_output_like: Whatever a traced task call just returned.

    Returns:
        That call's `AssembledTask.name`.
    """

    if isinstance(task_output_like, TaskOutput):
        return task_output_like.task_name

    if isinstance(task_output_like, dict):
        first = next(iter(task_output_like.values()))
        return first.task_name

    # A NamedTuple of `TaskOutput`s - index into it like the tuple it is.
    return task_output_like[0].task_name


def _place(task_output_like: T, compute_backend: BaseComputeBackend) -> T:
    """Places the task that produced `task_output_like` on `compute_backend`.

    Args:
        task_output_like: The just-called task's return value.
        compute_backend: The backend to place it on.

    Returns:
        `task_output_like`, unchanged - so pipeline code keeps composing
        (`return add(ab_on_batch, ab_on_lambda)`) exactly as if this
        wrapper weren't there.

    Raises:
        PipelineAssemblyError: If called outside an active pipeline trace,
            or `task_output_like` doesn't reference a task assembled in the
            active one.
    """

    context = get_active_context()
    if context is None:
        raise PipelineAssemblyError(
            f"{compute_backend.name}(...) can only be used while a "
            "pipeline is being traced, on a task call's own return value."
        )

    task_name = _task_name_of(task_output_like)
    if task_name not in context.assembled_tasks:
        raise PipelineAssemblyError(
            f"No assembled task named {task_name!r} found to place on "
            f"{compute_backend.name!r}."
        )

    context.assembled_tasks[task_name].compute_backend = compute_backend

    return task_output_like


def aws_batch(task_output_like: T, config: AwsBatchConfig) -> T:
    """Pins the task that produced `task_output_like` to run on AWS Batch.

    Usage::

        @pipeline
        def my_pipeline(a: int, b: int):
            ab = aws_batch(add(a, b), config=AwsBatchConfig(job_queue=...))
            return add(ab, b)

    Args:
        task_output_like: The task call to pin - its own return value,
            passed straight through.
        config: How to submit this task's job.

    Returns:
        `task_output_like`, unchanged.
    """

    return _place(task_output_like, AwsBatchComputeBackend(config))


def aws_lambda(task_output_like: T, config: AwsLambdaConfig) -> T:
    """Pins the task that produced `task_output_like` to run on AWS Lambda.

    Usage::

        @pipeline
        def my_pipeline(a: int, b: int):
            ab = aws_lambda(add(a, b), config=AwsLambdaConfig(function_name=...))
            return add(ab, b)

    Args:
        task_output_like: The task call to pin - its own return value,
            passed straight through.
        config: How to invoke this task's function.

    Returns:
        `task_output_like`, unchanged.
    """

    return _place(task_output_like, AwsLambdaComputeBackend(config))
