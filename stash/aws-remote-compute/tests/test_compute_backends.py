"""Unit tests for `AwsBatchComputeBackend`/`AwsLambdaComputeBackend`: pure
wiring/logic against a mocked `boto3.client`, matching
`test_s3_artifact_store.py`'s own convention - no real AWS.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from bettmensch_ai.pipelines.compute import (
    AwsBatchComputeBackend,
    AwsBatchConfig,
    AwsLambdaComputeBackend,
    AwsLambdaConfig,
)
from bettmensch_ai.pipelines.materializers import DefaultMaterializer
from bettmensch_ai.pipelines.runner.exceptions import (
    RemoteComputeConfigurationError,
    RemoteTaskExecutionError,
)
from bettmensch_ai.pipelines.task import task
from bettmensch_ai.pipelines.task.assembled_task import AssembledTask


@task
def add(a: int, b: int) -> int:
    return a + b


def make_assembled_task():
    return AssembledTask(
        name="add", task=add, static_inputs={"a": 1, "b": 2}, output_names=("result",)
    )


# --- AwsBatchComputeBackend ---------------------------------------------


def test_batch_run_requires_a_job_definition():
    backend = AwsBatchComputeBackend(AwsBatchConfig(job_queue="my-queue"))

    with pytest.raises(RemoteComputeConfigurationError):
        backend.run(
            make_assembled_task(),
            artifact_store=MagicMock(),
            code_bundle_key="bundle/key",
            input_keys={},
            output_keys={"result": "k/result"},
            pipeline_name="my-pipeline",
            pipeline_run_id="run-id",
            default_materializer_cls=DefaultMaterializer,
        )


@patch("bettmensch_ai.pipelines.compute.aws_batch_compute_backend.time.sleep")
@patch("bettmensch_ai.pipelines.compute.aws_batch_compute_backend.boto3.client")
def test_batch_run_polls_until_succeeded_and_returns_logs(mock_client, mock_sleep):
    batch_client = MagicMock()
    logs_client = MagicMock()
    mock_client.side_effect = lambda service, **kwargs: {
        "batch": batch_client,
        "logs": logs_client,
    }[service]

    batch_client.submit_job.return_value = {"jobId": "job-1"}
    batch_client.describe_jobs.side_effect = [
        {"jobs": [{"status": "RUNNING"}]},
        {
            "jobs": [
                {
                    "status": "SUCCEEDED",
                    "container": {"logStreamName": "stream-1"},
                }
            ]
        },
    ]
    logs_client.get_log_events.return_value = {"events": [{"message": "done"}]}

    backend = AwsBatchComputeBackend(
        AwsBatchConfig(job_queue="my-queue", job_definition="my-job-def")
    )
    logs = backend.run(
        make_assembled_task(),
        artifact_store=MagicMock(),
        code_bundle_key="bundle/key",
        input_keys={},
        output_keys={"result": "k/result"},
        pipeline_name="my-pipeline",
        pipeline_run_id="run-id",
        default_materializer_cls=DefaultMaterializer,
    )

    assert logs == "done"
    submit_kwargs = batch_client.submit_job.call_args.kwargs
    assert submit_kwargs["jobQueue"] == "my-queue"
    assert submit_kwargs["jobDefinition"] == "my-job-def"
    command = submit_kwargs["containerOverrides"]["command"]
    assert "--task-module" in command
    assert add.func.__module__ in command
    assert json.loads(command[command.index("--static-inputs") + 1]) == {"a": 1, "b": 2}
    assert mock_sleep.called


@patch("bettmensch_ai.pipelines.compute.aws_batch_compute_backend.time.sleep")
@patch("bettmensch_ai.pipelines.compute.aws_batch_compute_backend.boto3.client")
def test_batch_run_raises_on_failed_job(mock_client, mock_sleep):
    batch_client = MagicMock()
    logs_client = MagicMock()
    mock_client.side_effect = lambda service, **kwargs: {
        "batch": batch_client,
        "logs": logs_client,
    }[service]

    batch_client.submit_job.return_value = {"jobId": "job-1"}
    batch_client.describe_jobs.return_value = {
        "jobs": [
            {
                "status": "FAILED",
                "statusReason": "container exited with code 1",
                "container": {"logStreamName": "stream-1"},
            }
        ]
    }
    logs_client.get_log_events.return_value = {"events": [{"message": "boom"}]}

    backend = AwsBatchComputeBackend(
        AwsBatchConfig(job_queue="my-queue", job_definition="my-job-def")
    )

    with pytest.raises(RemoteTaskExecutionError, match="container exited with code 1"):
        backend.run(
            make_assembled_task(),
            artifact_store=MagicMock(),
            code_bundle_key="bundle/key",
            input_keys={},
            output_keys={"result": "k/result"},
            pipeline_name="my-pipeline",
            pipeline_run_id="run-id",
            default_materializer_cls=DefaultMaterializer,
        )


@patch("bettmensch_ai.pipelines.compute.aws_batch_compute_backend.boto3.client")
def test_batch_fetch_logs_never_raises_when_logs_are_unavailable(mock_client):
    backend = AwsBatchComputeBackend(
        AwsBatchConfig(job_queue="my-queue", job_definition="my-job-def")
    )

    assert backend._fetch_logs({"container": {}}) is None
    assert backend._fetch_logs({}) is None


# --- AwsLambdaComputeBackend ---------------------------------------------


def test_lambda_run_requires_a_function_name():
    backend = AwsLambdaComputeBackend(AwsLambdaConfig())

    with pytest.raises(RemoteComputeConfigurationError):
        backend.run(
            make_assembled_task(),
            artifact_store=MagicMock(),
            code_bundle_key="bundle/key",
            input_keys={},
            output_keys={"result": "k/result"},
            pipeline_name="my-pipeline",
            pipeline_run_id="run-id",
            default_materializer_cls=DefaultMaterializer,
        )


@patch("bettmensch_ai.pipelines.compute.aws_lambda_compute_backend.boto3.client")
def test_lambda_run_invokes_and_returns_logs(mock_client):
    lambda_client = MagicMock()
    mock_client.return_value = lambda_client
    payload = MagicMock()
    payload.read.return_value = json.dumps({"status": "succeeded", "logs": "hi"}).encode()
    lambda_client.invoke.return_value = {"Payload": payload}

    backend = AwsLambdaComputeBackend(AwsLambdaConfig(function_name="my-fn"))
    logs = backend.run(
        make_assembled_task(),
        artifact_store=MagicMock(),
        code_bundle_key="bundle/key",
        input_keys={},
        output_keys={"result": "k/result"},
        pipeline_name="my-pipeline",
        pipeline_run_id="run-id",
        default_materializer_cls=DefaultMaterializer,
    )

    assert logs == "hi"
    invoke_kwargs = lambda_client.invoke.call_args.kwargs
    assert invoke_kwargs["FunctionName"] == "my-fn"
    sent_payload = json.loads(invoke_kwargs["Payload"])
    assert sent_payload["static_inputs"] == {"a": 1, "b": 2}


@patch("bettmensch_ai.pipelines.compute.aws_lambda_compute_backend.boto3.client")
def test_lambda_run_raises_on_failed_invocation(mock_client):
    lambda_client = MagicMock()
    mock_client.return_value = lambda_client
    payload = MagicMock()
    payload.read.return_value = json.dumps(
        {"status": "failed", "traceback": "ValueError: boom"}
    ).encode()
    lambda_client.invoke.return_value = {"Payload": payload}

    backend = AwsLambdaComputeBackend(AwsLambdaConfig(function_name="my-fn"))

    with pytest.raises(RemoteTaskExecutionError, match="ValueError: boom"):
        backend.run(
            make_assembled_task(),
            artifact_store=MagicMock(),
            code_bundle_key="bundle/key",
            input_keys={},
            output_keys={"result": "k/result"},
            pipeline_name="my-pipeline",
            pipeline_run_id="run-id",
            default_materializer_cls=DefaultMaterializer,
        )
