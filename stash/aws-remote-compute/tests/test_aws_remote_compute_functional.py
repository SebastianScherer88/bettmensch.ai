"""AWS-gated functional test: ad-hoc `aws_batch()`/`aws_lambda()` execution
against a real, provisioned AWS stack (see `infrastructure/aws/README.md`) -
`add` runs as a real AWS Batch Fargate job, `multiply` as a real AWS Lambda
invocation, each against the job definition/function `infrastructure/aws/
test_fixtures.py` pre-creates specifically for this test.

Ad-hoc execution never provisions its own Batch job definition/Lambda
function (see `AwsBatchComputeBackend`/`AwsLambdaComputeBackend`'s own
docstrings - only `CompiledPipeline.register()` does that, exercised
instead by `test_aws_stepfunctions_functional.py`), so this test needs
something that already exists to run against, exactly like a real user
running against infrastructure someone registered earlier.

Skips entirely (via the `aws_stack_config` fixture in `tests/
conftest.py`) unless a real AWS stack's connection details are exported as
`BETTMENSCH_AI_AWS_TEST_*` environment variables and AWS is actually
reachable - never fails on a machine with no AWS credentials.
"""

from pathlib import Path

import pytest
from bettmensch_ai.pipelines.artifact_store import S3ArtifactStore, S3ArtifactStoreConfig
from bettmensch_ai.pipelines.compute import AwsBatchConfig, AwsLambdaConfig, aws_batch, aws_lambda
from bettmensch_ai.pipelines.materializers import resolve_materializer_from_artifact
from bettmensch_ai.pipelines.metadata_store import (
    PostgresMetadataStore,
    PostgresMetadataStoreConfig,
    RunStatus,
)
from bettmensch_ai.pipelines.pipeline import pipeline
from bettmensch_ai.pipelines.runner import LocalRunner
from bettmensch_ai.pipelines.task import task

pytestmark = [pytest.mark.functional, pytest.mark.aws]

# This test's own directory, not the repo root: `bettmensch_ai` itself is
# already baked into the Batch/Lambda task-runtime images (see
# docker/task-runtime/Dockerfile.{batch,lambda}'s own `COPY src`) - all
# `CodeBundler` needs to ship is this test module (`add`/`multiply` are
# defined here), which has no other way to reach the remote entrypoint.
_PROJECT_ROOT = Path(__file__).resolve().parent


@task
def add(a: int, b: int) -> int:
    return a + b


@task
def multiply(a: int, b: int) -> int:
    return a * b


def test_ad_hoc_remote_execution_across_batch_and_lambda(aws_stack_config):
    artifact_store = S3ArtifactStore(
        S3ArtifactStoreConfig(
            bucket=aws_stack_config.s3_bucket, region_name=aws_stack_config.region
        )
    )
    metadata_store = PostgresMetadataStore(
        PostgresMetadataStoreConfig(dsn=aws_stack_config.postgres_dsn)
    )

    @pipeline
    def ad_hoc_aws_pipeline(a: int, b: int, c: int):
        ab = aws_batch(
            add(a, b),
            config=AwsBatchConfig(
                job_queue=aws_stack_config.batch_job_queue,
                job_definition=aws_stack_config.batch_job_definition,
            ),
        )
        return aws_lambda(
            multiply(ab, c),
            config=AwsLambdaConfig(function_name=aws_stack_config.lambda_function_name),
        )

    runner = LocalRunner(artifact_store, metadata_store, project_root=_PROJECT_ROOT)

    result = runner.run(ad_hoc_aws_pipeline, a=1, b=2, c=3)

    assert result == 9  # (1 + 2) * 3, computed on real Batch then real Lambda

    # Most recent run of this pipeline name - a shared, persistent backend
    # may already hold earlier runs from previous invocations of this same
    # test (mirroring test_store_combinations_e2e.py's own convention).
    pipeline_runs = metadata_store.list_pipeline_runs("ad-hoc-aws-pipeline")
    assert len(pipeline_runs) >= 1
    pipeline_run = pipeline_runs[0]
    assert pipeline_run.status == RunStatus.SUCCEEDED

    task_runs = {
        r.task_name: r for r in metadata_store.list_task_runs(pipeline_run.pipeline_run_id)
    }
    assert set(task_runs) == {"add", "multiply"}
    assert all(r.status == RunStatus.SUCCEEDED for r in task_runs.values())

    # Cross-check: reload each task's real output from S3, using only the
    # key Postgres recorded for it - proving both real stores agree on what
    # actually happened on real Batch/Lambda, not just that `LocalRunner`
    # claims they do.
    add_outputs = metadata_store.list_task_outputs(pipeline_run.pipeline_run_id, "add")
    assert len(add_outputs) == 1
    add_key = add_outputs[0].artifact_key
    add_materializer = resolve_materializer_from_artifact(artifact_store, add_key)
    assert artifact_store.load(add_materializer, add_key) == 3

    multiply_outputs = metadata_store.list_task_outputs(
        pipeline_run.pipeline_run_id, "multiply"
    )
    assert len(multiply_outputs) == 1
    multiply_key = multiply_outputs[0].artifact_key
    multiply_materializer = resolve_materializer_from_artifact(artifact_store, multiply_key)
    assert artifact_store.load(multiply_materializer, multiply_key) == 9
