"""Stashed from `tests/conftest.py` - the `aws_stack_config` fixture (and its
supporting `AwsStackConfig`/`_AWS_TEST_ENV_VARS`), used by the stashed
`test_aws_*.py` functional tests. Needs `import os` and `from dataclasses
import dataclass` (already in the active conftest.py) to drop back in.
"""

from dataclasses import dataclass


@dataclass
class AwsStackConfig:
    """Everything `tests/functional/pipelines/test_aws_*.py` needs to run
    against a real, provisioned AWS stack (see `infrastructure/aws/README.md`) -
    a thin, dependency-free bundle of the `BETTMENSCH_AI_AWS_TEST_*`
    environment variables, mirroring `s3_config`'s own bundling of
    `BETTMENSCH_AI_TEST_S3_*` into one fixture value.
    """

    region: str
    s3_bucket: str
    postgres_dsn: str
    batch_job_queue: str
    batch_job_role_arn: str
    batch_execution_role_arn: str
    batch_image: str
    batch_job_definition: str
    lambda_role_arn: str
    lambda_image_uri: str
    lambda_function_name: str
    stepfunctions_role_arn: str


# Required for every AWS-gated test - if any is unset, there is no
# provisioned stack to point at, so the whole tier skips (see
# `aws_stack_config` below), the same way `postgres_dsn`/`s3_config` skip
# rather than fail when their own infrastructure isn't up.
_AWS_TEST_ENV_VARS = {
    "region": "BETTMENSCH_AI_AWS_TEST_REGION",
    "s3_bucket": "BETTMENSCH_AI_AWS_TEST_S3_BUCKET",
    "postgres_dsn": "BETTMENSCH_AI_AWS_TEST_POSTGRES_DSN",
    "batch_job_queue": "BETTMENSCH_AI_AWS_TEST_BATCH_JOB_QUEUE",
    "batch_job_role_arn": "BETTMENSCH_AI_AWS_TEST_BATCH_JOB_ROLE_ARN",
    "batch_execution_role_arn": "BETTMENSCH_AI_AWS_TEST_BATCH_EXECUTION_ROLE_ARN",
    "batch_image": "BETTMENSCH_AI_AWS_TEST_BATCH_IMAGE",
    "batch_job_definition": "BETTMENSCH_AI_AWS_TEST_BATCH_JOB_DEFINITION",
    "lambda_role_arn": "BETTMENSCH_AI_AWS_TEST_LAMBDA_ROLE_ARN",
    "lambda_image_uri": "BETTMENSCH_AI_AWS_TEST_LAMBDA_IMAGE_URI",
    "lambda_function_name": "BETTMENSCH_AI_AWS_TEST_LAMBDA_FUNCTION_NAME",
    "stepfunctions_role_arn": "BETTMENSCH_AI_AWS_TEST_STEPFUNCTIONS_ROLE_ARN",
}


@pytest.fixture(scope="session")
def aws_stack_config() -> AwsStackConfig:
    """The real, provisioned AWS stack's connection details (see
    `infrastructure/aws/README.md`), skipping if any required `BETTMENSCH_AI_AWS_
    TEST_*` variable is unset or if AWS itself isn't reachable with
    whatever credentials are available (a cheap `sts.get_caller_identity`
    probe, mirroring `s3_config`'s own short-timeout, single-attempt
    reachability check) - so this tier is silently absent, never failing,
    on a machine with no AWS credentials at all (true of this project's own
    dev/CI sandbox as of when this was written).
    """

    missing = [
        env_var for env_var in _AWS_TEST_ENV_VARS.values() if not os.environ.get(env_var)
    ]
    if missing:
        pytest.skip(
            f"AWS-gated tests need {missing} set - see infrastructure/aws/README.md's "
            "outputs table (provision the stack with `make aws.up`, then "
            "export its outputs)."
        )

    import boto3
    import botocore.exceptions
    from botocore.config import Config

    try:
        boto3.client(
            "sts",
            region_name=os.environ["BETTMENSCH_AI_AWS_TEST_REGION"],
            config=Config(connect_timeout=2, read_timeout=2, retries={"max_attempts": 1}),
        ).get_caller_identity()
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as exc:
        pytest.skip(f"No usable AWS credentials for the AWS-gated tests ({exc}).")

    return AwsStackConfig(
        **{
            field_name: os.environ[env_var]
            for field_name, env_var in _AWS_TEST_ENV_VARS.items()
        }
    )
