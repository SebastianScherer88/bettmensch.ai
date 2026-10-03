"""AWS Batch: a Fargate compute environment + one job queue.

Deliberately does *not* create any job definitions here - those are
dynamic, created by `CompiledPipeline.register()` (per-pipeline, per-task,
named `f"{pipeline_name}-{task_name}"`) or supplied directly to
`aws_batch(...)` for ad-hoc execution against something that already
exists. This module only provisions the durable scaffolding a job
definition submits *into*. See `test_fixtures.py` for the one job
definition this stack does pre-create, dedicated to the AWS-gated ad-hoc
execution test.

Fargate, not EC2: no compute capacity to size/manage - the simplest "basic"
choice (see docs/design-decisions.md). GPU-requiring tasks aren't
supported by this compute environment for the same reason.
"""

from dataclasses import dataclass

import pulumi_aws as aws

from network import Network


@dataclass
class Batch:
    job_queue_arn: str
    job_queue_name: str


def create_batch(network: Network) -> Batch:
    compute_environment = aws.batch.ComputeEnvironment(
        "bettmensch-ai-batch",
        type="MANAGED",
        compute_resources=aws.batch.ComputeEnvironmentComputeResourcesArgs(
            type="FARGATE",
            max_vcpus=16,
            subnets=network.subnet_ids,
            security_group_ids=[network.compute_security_group_id],
        ),
        tags={"bettmensch-ai": "true"},
    )

    job_queue = aws.batch.JobQueue(
        "bettmensch-ai-batch",
        state="ENABLED",
        priority=1,
        compute_environment_orders=[
            aws.batch.JobQueueComputeEnvironmentOrderArgs(
                order=1, compute_environment=compute_environment.arn
            )
        ],
        tags={"bettmensch-ai": "true"},
    )

    return Batch(job_queue_arn=job_queue.arn, job_queue_name=job_queue.name)
