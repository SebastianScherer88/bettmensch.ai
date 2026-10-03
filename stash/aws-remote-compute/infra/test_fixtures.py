"""One pre-created Batch job definition + Lambda function, dedicated to
`tests/functional/pipelines/test_aws_remote_compute_functional.py`.

`aws_batch(...)`/`aws_lambda(...)` never provision anything for *ad-hoc*
execution (see `AwsBatchComputeBackend`/`AwsLambdaComputeBackend`'s own
docstrings - only `CompiledPipeline.register()` does that, dynamically) -
so exercising the ad-hoc path for real needs something that already exists
to run against, exactly like a real user would have from a prior
registration. This module is that "prior registration," fixed and owned by
this stack rather than created/torn down by the test itself, so the test
only has to *use* infrastructure, not manage it - `test_aws_stepfunctions_
functional.py` is the one that exercises real create/register/deregister.

Both reference `<repo>:latest` images that must already be pushed (see
`infrastructure/aws/README.md`) before this stack's first `pulumi up` succeeds -
unlike the frontend service (which merely fails to start healthy tasks
until pushed), `register_job_definition`/`create_function` fail outright
against a nonexistent image, so image pushes must happen first for this
module specifically.
"""

import json
from dataclasses import dataclass

import pulumi
import pulumi_aws as aws

from iam import Iam
from network import Network
from registry import Registry
from storage import Storage

_TAGS = {"bettmensch-ai": "true", "purpose": "test"}


@dataclass
class TestFixtures:
    batch_job_definition: pulumi.Output[str]
    lambda_function_name: pulumi.Output[str]


def create_test_fixtures(
    network: Network, storage: Storage, registry: Registry, iam: Iam
) -> TestFixtures:
    environment = pulumi.Output.all(storage.bucket_name).apply(
        lambda args: [{"name": "BETTMENSCH_AI_S3_ARTIFACT_STORE_BUCKET", "value": args[0]}]
    )

    job_definition = aws.batch.JobDefinition(
        "bettmensch-ai-test-ad-hoc",
        type="container",
        platform_capabilities=["FARGATE"],
        container_properties=pulumi.Output.all(
            registry.batch_task_runtime_repo_url,
            iam.batch_job_role_arn,
            iam.batch_execution_role_arn,
            environment,
        ).apply(
            lambda args: json.dumps(
                {
                    "image": f"{args[0]}:latest",
                    "jobRoleArn": args[1],
                    "executionRoleArn": args[2],
                    "environment": args[3],
                    "resourceRequirements": [
                        {"type": "VCPU", "value": "0.25"},
                        {"type": "MEMORY", "value": "512"},
                    ],
                    "networkConfiguration": {"assignPublicIp": "ENABLED"},
                }
            )
        ),
        tags=_TAGS,
    )

    lambda_function = aws.lambda_.Function(
        "bettmensch-ai-test-ad-hoc",
        package_type="Image",
        image_uri=registry.lambda_task_runtime_repo_url.apply(lambda url: f"{url}:latest"),
        role=iam.lambda_role_arn,
        timeout=60,
        environment=aws.lambda_.FunctionEnvironmentArgs(
            variables={"BETTMENSCH_AI_S3_ARTIFACT_STORE_BUCKET": storage.bucket_name}
        ),
        tags=_TAGS,
    )

    return TestFixtures(
        batch_job_definition=job_definition.arn,
        # `.name`, not `.function_name` - that's the Pulumi *input* property
        # name; the resource's own output for its (possibly
        # Pulumi-auto-suffixed) actual name is `.name`.
        lambda_function_name=lambda_function.name,
    )
