"""ECS: a Fargate cluster/service running `docker/frontend`'s image, wired
to the real S3 bucket/RDS instance this stack provisions - the genuine-AWS
counterpart of `pipelines.docker-compose.yaml`'s `frontend` service.

No ALB in this "basic" stack (see docs/design-decisions.md): the service
gets a public IP directly. The task references `<repo>:latest` before any
image has actually been pushed - the service simply won't start healthy
tasks until `make frontend.push.ecr` (see `infrastructure/aws/README.md`) has pushed
one; that ordering (provision first, push second) is expected, not a bug.
"""

import json
from dataclasses import dataclass

import pulumi
import pulumi_aws as aws

from iam import Iam
from network import Network
from registry import Registry
from storage import Storage

_CONTAINER_PORT = 8080


def create_frontend(network: Network, storage: Storage, registry: Registry, iam: Iam):
    cluster = aws.ecs.Cluster("bettmensch-ai", tags={"bettmensch-ai": "true"})

    log_group = aws.cloudwatch.LogGroup(
        "bettmensch-ai-frontend", retention_in_days=7, tags={"bettmensch-ai": "true"}
    )

    # Resolved once, up front - not from inside the `.apply()` below. `aws.
    # get_region()` is a data-source *invoke*, a different synchronization
    # path from a resource Output; calling it from inside an `.apply()`
    # callback raced against the engine's gRPC connection tearing down
    # whenever anything else in the same `pulumi up` failed concurrently
    # (observed directly: "grpc: the client connection is closing", while
    # a separate Lambda function resource failed elsewhere in the same
    # update). The region itself never depends on any other resource's
    # output, so there's no reason to defer resolving it at all.
    region_name = aws.get_region().name

    container_definitions = pulumi.Output.all(
        registry.frontend_repo_url, storage.dsn, storage.bucket_name, log_group.name
    ).apply(
        lambda args: json.dumps(
            [
                {
                    "name": "frontend",
                    "image": f"{args[0]}:latest",
                    "portMappings": [{"containerPort": _CONTAINER_PORT, "protocol": "tcp"}],
                    "environment": [
                        {
                            "name": "BETTMENSCH_AI_POSTGRES_METADATA_STORE_DSN",
                            "value": args[1],
                        },
                        {"name": "BETTMENSCH_AI_S3_ARTIFACT_STORE_BUCKET", "value": args[2]},
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": args[3],
                            "awslogs-region": region_name,
                            "awslogs-stream-prefix": "frontend",
                        },
                    },
                }
            ]
        )
    )

    task_definition = aws.ecs.TaskDefinition(
        "bettmensch-ai-frontend",
        family="bettmensch-ai-frontend",
        cpu="256",
        memory="512",
        network_mode="awsvpc",
        requires_compatibilities=["FARGATE"],
        execution_role_arn=iam.ecs_task_execution_role_arn,
        task_role_arn=iam.ecs_task_role_arn,
        container_definitions=container_definitions,
        tags={"bettmensch-ai": "true"},
    )

    service = aws.ecs.Service(
        "bettmensch-ai-frontend",
        cluster=cluster.arn,
        task_definition=task_definition.arn,
        desired_count=1,
        launch_type="FARGATE",
        network_configuration=aws.ecs.ServiceNetworkConfigurationArgs(
            subnets=network.subnet_ids,
            security_groups=[network.ecs_security_group_id],
            assign_public_ip=True,
        ),
        tags={"bettmensch-ai": "true"},
    )

    return cluster, service
