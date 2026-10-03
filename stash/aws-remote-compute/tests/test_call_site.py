from typing import NamedTuple, TypedDict

import pytest
from bettmensch_ai.pipelines.compute import (
    AwsBatchComputeBackend,
    AwsBatchConfig,
    AwsLambdaComputeBackend,
    AwsLambdaConfig,
    aws_batch,
    aws_lambda,
)
from bettmensch_ai.pipelines.context import PipelineAssemblyContext, PipelineAssemblyError
from bettmensch_ai.pipelines.task import task


@task
def add(a: int, b: int) -> int:
    return a + b


def test_aws_batch_places_the_task_on_a_batch_backend():
    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {}
        aws_batch(add(1, 2), config=AwsBatchConfig(job_queue="my-queue"))

    assembled_task = context.assembled_tasks["add"]
    assert isinstance(assembled_task.compute_backend, AwsBatchComputeBackend)
    assert assembled_task.compute_backend.config.job_queue == "my-queue"


def test_aws_lambda_places_the_task_on_a_lambda_backend():
    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {}
        aws_lambda(add(1, 2), config=AwsLambdaConfig(function_name="my-fn"))

    assembled_task = context.assembled_tasks["add"]
    assert isinstance(assembled_task.compute_backend, AwsLambdaComputeBackend)
    assert assembled_task.compute_backend.config.function_name == "my-fn"


def test_placement_wrapper_returns_the_task_output_unchanged():
    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {}
        original = add(1, 2)
        placed = aws_batch(original, config=AwsBatchConfig(job_queue="my-queue"))

    assert placed == original


class DivModOutput(NamedTuple):
    quotient: int
    remainder: int


@task
def divmod_task(a: int, b: int) -> DivModOutput:
    return DivModOutput(quotient=a // b, remainder=a % b)


def test_placement_wrapper_handles_named_tuple_output():
    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {}
        output = aws_batch(divmod_task(7, 2), config=AwsBatchConfig(job_queue="q"))

    assert isinstance(context.assembled_tasks["divmod-task"].compute_backend, AwsBatchComputeBackend)
    assert output.quotient.task_name == "divmod-task"


class DivModDict(TypedDict):
    quotient: int
    remainder: int


@task
def divmod_dict_task(a: int, b: int) -> DivModDict:
    return {"quotient": a // b, "remainder": a % b}


def test_placement_wrapper_handles_typed_dict_output():
    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {}
        output = aws_batch(divmod_dict_task(7, 2), config=AwsBatchConfig(job_queue="q"))

    assert isinstance(
        context.assembled_tasks["divmod-dict-task"].compute_backend, AwsBatchComputeBackend
    )
    assert output["quotient"].task_name == "divmod-dict-task"


def test_placement_wrapper_raises_outside_an_active_trace():
    with pytest.raises(PipelineAssemblyError):
        aws_batch(add(1, 2), config=AwsBatchConfig(job_queue="q"))
