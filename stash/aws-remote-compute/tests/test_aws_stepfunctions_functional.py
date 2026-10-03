"""AWS-gated functional test: the full dynamic
`StepFunctionsCompiler.compile -> CompiledPipeline.register -> RemoteRunner.
run -> CompiledPipeline.deregister` lifecycle against a real, provisioned
AWS stack (see `infrastructure/aws/README.md`) - a genuine Step Functions
state machine execution, not a mocked `boto3.client`, unlike every other
test of this lifecycle in this project.

Unlike `test_aws_remote_compute_functional.py` (which runs against
pre-existing, `infrastructure/aws`-provisioned job definition/function),
this test's whole point is exercising the *dynamic* creation path: both
tasks' configs carry only registration-time fields (`image`/`job_role_arn`/
`execution_role_arn`, `image_uri`/`role_arn`), so `CompiledPipeline.
register()` really does create a fresh Batch job definition, a fresh Lambda
function, and a fresh state machine - and `deregister()` really does delete
all three afterwards, proving "full cleanup on deregister" against real AWS
too, not just the existing mocked integration test
(`test_stepfunctions_compiler_integration.py`).

Skips entirely (via the `aws_stack_config` fixture) unless a real AWS
stack's connection details are exported as `BETTMENSCH_AI_AWS_TEST_*`
environment variables and AWS is actually reachable.
"""

from pathlib import Path

import pytest
from bettmensch_ai.pipelines.artifact_store import S3ArtifactStore, S3ArtifactStoreConfig
from bettmensch_ai.pipelines.compilers.stepfunctions import StepFunctionsCompiler
from bettmensch_ai.pipelines.compute import AwsBatchConfig, AwsLambdaConfig, aws_batch, aws_lambda
from bettmensch_ai.pipelines.materializers import resolve_materializer_from_artifact
from bettmensch_ai.pipelines.metadata_store import (
    PostgresMetadataStore,
    PostgresMetadataStoreConfig,
    RunStatus,
)
from bettmensch_ai.pipelines.pipeline import pipeline
from bettmensch_ai.pipelines.runner import RegisteredPipeline, RemoteRunner
from bettmensch_ai.pipelines.task import task

pytestmark = [pytest.mark.functional, pytest.mark.aws]

# This test's own directory - see test_aws_remote_compute_functional.py's
# own comment on why.
_PROJECT_ROOT = Path(__file__).resolve().parent


@task
def add(a: int, b: int) -> int:
    return a + b


@task
def multiply(a: int, b: int) -> int:
    return a * b


def test_register_run_and_deregister_against_real_stepfunctions(
    aws_stack_config, unique_key_prefix
):
    artifact_store = S3ArtifactStore(
        S3ArtifactStoreConfig(
            bucket=aws_stack_config.s3_bucket, region_name=aws_stack_config.region
        )
    )
    metadata_store = PostgresMetadataStore(
        PostgresMetadataStoreConfig(dsn=aws_stack_config.postgres_dsn)
    )

    # A fresh registration-scoped environment for the dynamically-created
    # job definition/function to find their own artifact store - see
    # AwsBatchConfig/AwsLambdaConfig's `environment` field.
    store_environment = {
        "BETTMENSCH_AI_S3_ARTIFACT_STORE_BUCKET": aws_stack_config.s3_bucket,
        "BETTMENSCH_AI_S3_ARTIFACT_STORE_REGION_NAME": aws_stack_config.region,
    }

    # Unique per test invocation: a leftover resource from an earlier failed
    # run (e.g. a crashed `deregister()`) must not collide with this one -
    # unlike test_store_combinations_e2e.py's fixed pipeline name, real AWS
    # resources (a Lambda function name, in particular) aren't disposable
    # the way a fresh tmp_path/SQLite file is.
    pipeline_name = f"aws-sfn-{unique_key_prefix}"

    @pipeline(name=pipeline_name)
    def aws_sfn_pipeline(a: int, b: int, c: int):
        ab = aws_batch(
            add(a, b),
            config=AwsBatchConfig(
                job_queue=aws_stack_config.batch_job_queue,
                image=aws_stack_config.batch_image,
                job_role_arn=aws_stack_config.batch_job_role_arn,
                execution_role_arn=aws_stack_config.batch_execution_role_arn,
                environment=store_environment,
            ),
        )
        return aws_lambda(
            multiply(ab, c),
            config=AwsLambdaConfig(
                image_uri=aws_stack_config.lambda_image_uri,
                role_arn=aws_stack_config.lambda_role_arn,
                environment=store_environment,
            ),
        )

    compiled = StepFunctionsCompiler().compile(aws_sfn_pipeline)
    registration_id = compiled.register(
        metadata_store, state_machine_role_arn=aws_stack_config.stepfunctions_role_arn
    )

    try:
        registered = RegisteredPipeline.from_registration(metadata_store, registration_id)
        assert registered.backend == "aws_stepfunctions"
        # register() really did create fresh, dynamic AWS resources
        assert registered.backend_metadata["state_machine_arn"]
        assert registered.backend_metadata["batch_job_definitions"]["add"]
        assert registered.backend_metadata["lambda_functions"]["multiply"]

        result = RemoteRunner(
            artifact_store, metadata_store, project_root=_PROJECT_ROOT
        ).run(registered, a=1, b=2, c=3)

        assert result == 9  # (1 + 2) * 3, computed on a real Step Functions execution

        pipeline_runs = metadata_store.list_pipeline_runs(pipeline_name)
        assert len(pipeline_runs) == 1
        pipeline_run = pipeline_runs[0]
        assert pipeline_run.status == RunStatus.SUCCEEDED
        assert pipeline_run.pipeline_assembly_id is not None

        task_runs = {
            r.task_name: r
            for r in metadata_store.list_task_runs(pipeline_run.pipeline_run_id)
        }
        assert set(task_runs) == {"add", "multiply"}
        assert all(r.status == RunStatus.SUCCEEDED for r in task_runs.values())

        multiply_outputs = metadata_store.list_task_outputs(
            pipeline_run.pipeline_run_id, "multiply"
        )
        assert len(multiply_outputs) == 1
        multiply_key = multiply_outputs[0].artifact_key
        multiply_materializer = resolve_materializer_from_artifact(
            artifact_store, multiply_key
        )
        assert artifact_store.load(multiply_materializer, multiply_key) == 9
    finally:
        # Full cleanup, against real AWS - proves deregister() actually
        # deletes everything register() created, not just marks the
        # registration inactive in Postgres.
        compiled.deregister(metadata_store, registration_id)

    registration = metadata_store.get_pipeline_registration(registration_id)
    assert registration.is_active is False
