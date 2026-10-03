"""Tests `CompiledPipeline.register()`/`.deregister()` against a mocked
`boto3` (no real AWS available in this environment - see
`test_compute_backends.py`'s own convention) but a real `LocalMetadataStore`,
so the full register -> metadata_store.register_pipeline -> deregister ->
metadata_store.deregister_pipeline round trip is exercised end to end.
"""

from unittest.mock import MagicMock, patch

from bettmensch_ai.pipelines.compilers.stepfunctions import StepFunctionsCompiler
from bettmensch_ai.pipelines.compute import AwsBatchConfig, AwsLambdaConfig, aws_batch, aws_lambda
from bettmensch_ai.pipelines.metadata_store import LocalMetadataStore, LocalMetadataStoreConfig
from bettmensch_ai.pipelines.pipeline import pipeline
from bettmensch_ai.pipelines.task import task


@task
def add(a: int, b: int) -> int:
    return a + b


@task
def multiply(a: int, b: int) -> int:
    return a * b


def make_metadata_store(tmp_path):
    return LocalMetadataStore(LocalMetadataStoreConfig(db_path=str(tmp_path / "metadata.db")))


@patch("bettmensch_ai.pipelines.compilers.stepfunctions.compiled_pipeline.boto3.client")
def test_register_creates_missing_job_definition_and_state_machine(mock_client, tmp_path):
    metadata_store = make_metadata_store(tmp_path)

    stepfunctions_client = MagicMock()
    batch_client = MagicMock()
    mock_client.side_effect = lambda service, **kw: {
        "stepfunctions": stepfunctions_client,
        "batch": batch_client,
        "lambda": MagicMock(),
    }[service]

    batch_client.register_job_definition.return_value = {
        "jobDefinitionArn": "arn:aws:batch:job-definition/my-pipeline-add:1",
        "revision": 1,
    }
    stepfunctions_client.create_state_machine.return_value = {
        "stateMachineArn": "arn:aws:states:::stateMachine:my-pipeline"
    }

    @pipeline
    def my_pipeline(a: int, b: int):
        return aws_batch(
            add(a, b),
            config=AwsBatchConfig(
                job_queue="my-queue",
                image="my-image:latest",
                job_role_arn="arn:aws:iam::123:role/job",
                execution_role_arn="arn:aws:iam::123:role/exec",
            ),
        )

    compiled = StepFunctionsCompiler().compile(my_pipeline)
    registration_id = compiled.register(
        metadata_store, state_machine_role_arn="arn:aws:iam::123:role/states"
    )

    # job definition created under the deterministic pipeline-task name, tagged,
    # and Fargate-valid (platform capability, non-empty resource requirements,
    # a public-IP network config - this stack's Batch compute environment has
    # no NAT gateway)
    register_kwargs = batch_client.register_job_definition.call_args.kwargs
    assert register_kwargs["jobDefinitionName"] == "my-pipeline-add"
    assert register_kwargs["tags"] == {"bettmensch-ai-pipeline": "my-pipeline", "bettmensch-ai-task": "add"}
    assert register_kwargs["platformCapabilities"] == ["FARGATE"]
    resource_requirements = register_kwargs["containerProperties"]["resourceRequirements"]
    assert {"type": "VCPU", "value": "0.25"} in resource_requirements
    assert {"type": "MEMORY", "value": "512"} in resource_requirements
    assert register_kwargs["containerProperties"]["networkConfiguration"] == {
        "assignPublicIp": "ENABLED"
    }

    # state machine created and tagged, with the role Step Functions itself assumes
    create_kwargs = stepfunctions_client.create_state_machine.call_args.kwargs
    assert create_kwargs["name"] == "my-pipeline"
    assert create_kwargs["roleArn"] == "arn:aws:iam::123:role/states"
    assert {"key": "bettmensch-ai-pipeline", "value": "my-pipeline"} in create_kwargs["tags"]

    # the finalized ASL references the job definition just created
    assert "my-pipeline-add:1" in create_kwargs["definition"]

    registration = metadata_store.get_pipeline_registration(registration_id)
    assert registration.backend == "aws_stepfunctions"
    assert registration.backend_metadata["state_machine_arn"] == "arn:aws:states:::stateMachine:my-pipeline"
    assert registration.backend_metadata["batch_job_definitions"] == {
        "add": "arn:aws:batch:job-definition/my-pipeline-add:1"
    }
    assert registration.backend_metadata["lambda_functions"] == {}


@patch("bettmensch_ai.pipelines.compilers.stepfunctions.compiled_pipeline.boto3.client")
def test_register_reuses_an_existing_job_definition(mock_client, tmp_path):
    metadata_store = make_metadata_store(tmp_path)

    stepfunctions_client = MagicMock()
    batch_client = MagicMock()
    mock_client.side_effect = lambda service, **kw: {
        "stepfunctions": stepfunctions_client,
        "batch": batch_client,
    }[service]
    stepfunctions_client.create_state_machine.return_value = {"stateMachineArn": "arn:...:sm"}

    @pipeline
    def my_pipeline(a: int, b: int):
        return aws_batch(
            add(a, b), config=AwsBatchConfig(job_queue="q", job_definition="already-exists:3")
        )

    compiled = StepFunctionsCompiler().compile(my_pipeline)
    registration_id = compiled.register(
        metadata_store, state_machine_role_arn="arn:aws:iam::123:role/states"
    )

    batch_client.register_job_definition.assert_not_called()
    registration = metadata_store.get_pipeline_registration(registration_id)
    assert registration.backend_metadata["batch_job_definitions"] == {}


@patch("bettmensch_ai.pipelines.compilers.stepfunctions.compiled_pipeline.boto3.client")
def test_deregister_deletes_every_resource_register_created(mock_client, tmp_path):
    metadata_store = make_metadata_store(tmp_path)

    stepfunctions_client = MagicMock()
    batch_client = MagicMock()
    lambda_client = MagicMock()
    mock_client.side_effect = lambda service, **kw: {
        "stepfunctions": stepfunctions_client,
        "batch": batch_client,
        "lambda": lambda_client,
    }[service]

    batch_client.register_job_definition.return_value = {
        "jobDefinitionArn": "arn:aws:batch:job-definition/my-pipeline-add:1",
        "revision": 1,
    }
    lambda_client.create_function.return_value = {"FunctionArn": "arn:aws:lambda:my-pipeline-multiply"}
    stepfunctions_client.create_state_machine.return_value = {"stateMachineArn": "arn:...:sm"}

    @pipeline
    def my_pipeline(a: int, b: int, c: int):
        ab = aws_batch(
            add(a, b),
            config=AwsBatchConfig(
                job_queue="q", image="img", job_role_arn="r1", execution_role_arn="r2"
            ),
        )
        return aws_lambda(
            multiply(ab, c),
            config=AwsLambdaConfig(
                image_uri="img", role_arn="r3", environment={"BUCKET": "my-bucket"}
            ),
        )

    compiled = StepFunctionsCompiler().compile(my_pipeline)
    registration_id = compiled.register(
        metadata_store, state_machine_role_arn="arn:aws:iam::123:role/states"
    )

    # the function's environment carries the registration-time config through
    assert lambda_client.create_function.call_args.kwargs["Environment"] == {
        "Variables": {"BUCKET": "my-bucket"}
    }

    compiled.deregister(metadata_store, registration_id)

    stepfunctions_client.delete_state_machine.assert_called_once_with(
        stateMachineArn="arn:...:sm"
    )
    batch_client.deregister_job_definition.assert_called_once_with(
        jobDefinition="arn:aws:batch:job-definition/my-pipeline-add:1"
    )
    lambda_client.delete_function.assert_called_once_with(
        FunctionName="arn:aws:lambda:my-pipeline-multiply"
    )

    registration = metadata_store.get_pipeline_registration(registration_id)
    assert registration.is_active is False
