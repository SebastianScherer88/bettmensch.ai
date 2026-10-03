import pytest
from bettmensch_ai.pipelines.compilers.exceptions import UnsupportedComputeBackendError
from bettmensch_ai.pipelines.compilers.stepfunctions import CompiledPipeline, StepFunctionsCompiler
from bettmensch_ai.pipelines.compute import AwsBatchConfig, AwsLambdaConfig, aws_batch, aws_lambda
from bettmensch_ai.pipelines.pipeline import pipeline
from bettmensch_ai.pipelines.task import task


@task
def add(a: int, b: int) -> int:
    return a + b


@task
def multiply(a: int, b: int) -> int:
    return a * b


def test_compile_rejects_a_pipeline_with_a_local_task():
    @pipeline
    def my_pipeline(a: int, b: int):
        return add(a, b)

    with pytest.raises(UnsupportedComputeBackendError, match="add"):
        StepFunctionsCompiler().compile(my_pipeline)


def test_compile_produces_sequential_states_for_a_linear_pipeline():
    @pipeline
    def my_pipeline(a: int, b: int, c: int):
        ab = aws_batch(add(a, b), config=AwsBatchConfig(job_queue="q", job_definition="jd"))
        return aws_batch(multiply(ab, c), config=AwsBatchConfig(job_queue="q", job_definition="jd2"))

    compiled = StepFunctionsCompiler().compile(my_pipeline)

    assert isinstance(compiled, CompiledPipeline)
    assert compiled.definition["StartAt"] == "add"
    assert compiled.definition["States"]["add"]["Next"] == "multiply"
    assert compiled.definition["States"]["multiply"]["End"] is True
    assert "Next" not in compiled.definition["States"]["multiply"]


def test_compile_wires_batch_task_parameters():
    @pipeline
    def my_pipeline(a: int, b: int):
        return aws_batch(add(a, b), config=AwsBatchConfig(job_queue="my-queue", job_definition="my-jd"))

    compiled = StepFunctionsCompiler().compile(my_pipeline)
    state = compiled.definition["States"]["add"]

    assert state["Type"] == "Task"
    assert state["Resource"] == "arn:aws:states:::batch:submitJob.sync"
    assert state["Parameters"]["JobQueue"] == "my-queue"
    assert state["Parameters"]["JobDefinition"] == "my-jd"
    assert state["ResultPath"] == "$.add"


def test_compile_wires_lambda_task_parameters():
    @pipeline
    def my_pipeline(a: int, b: int):
        return aws_lambda(add(a, b), config=AwsLambdaConfig(function_name="my-fn"))

    compiled = StepFunctionsCompiler().compile(my_pipeline)
    state = compiled.definition["States"]["add"]

    assert state["Type"] == "Task"
    assert state["Resource"] == "arn:aws:states:::lambda:invoke"
    assert state["Parameters"]["FunctionName"] == "my-fn"
    assert state["Parameters"]["Payload"]["task_qualname"] == "add"


def test_compile_uses_a_parallel_state_for_a_multi_task_rank():
    @task
    def subtract(a: int, b: int) -> int:
        return a - b

    @pipeline
    def my_pipeline(a: int, b: int):
        aws_batch(add(a, b), config=AwsBatchConfig(job_queue="q", job_definition="jd"))
        return aws_batch(subtract(a, b), config=AwsBatchConfig(job_queue="q", job_definition="jd"))

    compiled = StepFunctionsCompiler().compile(my_pipeline)
    start_state = compiled.definition["States"][compiled.definition["StartAt"]]

    assert start_state["Type"] == "Parallel"
    branch_starts = {branch["StartAt"] for branch in start_state["Branches"]}
    assert branch_starts == {"add", "subtract"}
