"""Stashed from `infrastructure/aws/iam.py` - the IAM roles for Batch jobs,
Lambda functions, and Step Functions state machines. Re-add these resources
(and the corresponding `Iam` dataclass fields/outputs) to the active
`iam.py`/`__main__.py` when remote compute comes back. The active `iam.py`
still defines `_ECS_TASKS_TRUST_POLICY`; `_LAMBDA_TRUST_POLICY`/
`_STATES_TRUST_POLICY`/`_s3_read_write_policy` below were trimmed from it
since nothing active used them once this moved - self-contained here.
"""

import json

import pulumi
import pulumi_aws as aws

_LAMBDA_TRUST_POLICY = json.dumps(
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
)
_STATES_TRUST_POLICY = json.dumps(
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "states.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
)


def _s3_read_write_policy(bucket_arn: pulumi.Output[str]) -> pulumi.Output[str]:
    return bucket_arn.apply(
        lambda arn: json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                        "Resource": [arn, f"{arn}/*"],
                    }
                ],
            }
        )
    )


def _create_remote_compute_iam(storage):
    # --- Batch (Fargate) -----------------------------------------------
    batch_job_role = aws.iam.Role(
        "bettmensch-ai-batch-job",
        assume_role_policy=_ECS_TASKS_TRUST_POLICY,
        tags={"bettmensch-ai": "true"},
    )
    aws.iam.RolePolicy(
        "bettmensch-ai-batch-job-s3",
        role=batch_job_role.id,
        policy=_s3_read_write_policy(storage.bucket_arn),
    )

    batch_execution_role = aws.iam.Role(
        "bettmensch-ai-batch-execution",
        assume_role_policy=_ECS_TASKS_TRUST_POLICY,
        tags={"bettmensch-ai": "true"},
    )
    aws.iam.RolePolicyAttachment(
        "bettmensch-ai-batch-execution-managed",
        role=batch_execution_role.name,
        policy_arn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
    )

    # --- Lambda -----------------------------------------------------------
    lambda_role = aws.iam.Role(
        "bettmensch-ai-lambda",
        assume_role_policy=_LAMBDA_TRUST_POLICY,
        tags={"bettmensch-ai": "true"},
    )
    aws.iam.RolePolicyAttachment(
        "bettmensch-ai-lambda-managed",
        role=lambda_role.name,
        policy_arn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    )
    aws.iam.RolePolicy(
        "bettmensch-ai-lambda-s3",
        role=lambda_role.id,
        policy=_s3_read_write_policy(storage.bucket_arn),
    )

    # --- Step Functions -----------------------------------------------
    stepfunctions_role = aws.iam.Role(
        "bettmensch-ai-stepfunctions",
        assume_role_policy=_STATES_TRUST_POLICY,
        tags={"bettmensch-ai": "true"},
    )
    aws.iam.RolePolicy(
        "bettmensch-ai-stepfunctions-invoke",
        role=stepfunctions_role.id,
        policy=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "batch:SubmitJob",
                            "batch:DescribeJobs",
                            "batch:TerminateJob",
                        ],
                        "Resource": "*",
                    },
                    {
                        "Effect": "Allow",
                        "Action": "lambda:InvokeFunction",
                        "Resource": "*",
                    },
                    {
                        # Required for the `.sync` Batch service integration:
                        # Step Functions manages an EventBridge rule to learn
                        # when the Batch job actually finishes.
                        "Effect": "Allow",
                        "Action": [
                            "events:PutTargets",
                            "events:PutRule",
                            "events:DescribeRule",
                        ],
                        "Resource": (
                            "arn:aws:events:*:*:rule/StepFunctionsGetEventsForBatchJobsRule"
                        ),
                    },
                ],
            }
        ),
    )

    return {
        "batch_job_role_arn": batch_job_role.arn,
        "batch_execution_role_arn": batch_execution_role.arn,
        "lambda_role_arn": lambda_role.arn,
        "stepfunctions_role_arn": stepfunctions_role.arn,
    }
