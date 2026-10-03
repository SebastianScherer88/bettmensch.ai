"""IAM: the ECS task roles the frontend service assumes.

Least-privilege for what this stack actually needs - the task role is
scoped to exactly the bucket this stack creates, not `*`.

The IAM roles for Batch jobs/Lambda functions/Step Functions state machines
have been stashed along with the rest of that remote-compute layer - see
`stash/aws-remote-compute/README.md`.
"""

import json
from dataclasses import dataclass

import pulumi
import pulumi_aws as aws

from storage import Storage

_ECS_TASKS_TRUST_POLICY = json.dumps(
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
)


@dataclass
class Iam:
    ecs_task_execution_role_arn: str
    ecs_task_role_arn: str


def _s3_read_policy(bucket_arn: pulumi.Output[str]) -> pulumi.Output[str]:
    return bucket_arn.apply(
        lambda arn: json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["s3:GetObject", "s3:ListBucket"],
                        "Resource": [arn, f"{arn}/*"],
                    }
                ],
            }
        )
    )


def create_iam(storage: Storage) -> Iam:
    ecs_task_execution_role = aws.iam.Role(
        "bettmensch-ai-ecs-execution",
        assume_role_policy=_ECS_TASKS_TRUST_POLICY,
        tags={"bettmensch-ai": "true"},
    )
    aws.iam.RolePolicyAttachment(
        "bettmensch-ai-ecs-execution-managed",
        role=ecs_task_execution_role.name,
        policy_arn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
    )

    ecs_task_role = aws.iam.Role(
        "bettmensch-ai-ecs-task",
        assume_role_policy=_ECS_TASKS_TRUST_POLICY,
        tags={"bettmensch-ai": "true"},
    )
    aws.iam.RolePolicy(
        "bettmensch-ai-ecs-task-s3",
        role=ecs_task_role.id,
        policy=_s3_read_policy(storage.bucket_arn),
    )

    return Iam(
        ecs_task_execution_role_arn=ecs_task_execution_role.arn,
        ecs_task_role_arn=ecs_task_role.arn,
    )
