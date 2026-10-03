"""Networking: the account's default VPC/subnets (no new VPC, no NAT
gateway - see docs/design-decisions.md for why this "basic" stack accepts
that trade-off) plus the security groups RDS/ECS need.

AWS Batch's Fargate compute environment and the dynamically-created Lambda
functions need no inbound security group at all (they only ever make
outbound calls to ECR/S3/RDS/CloudWatch), so this module only defines the
two groups that need to accept inbound traffic.
"""

from dataclasses import dataclass
from typing import List

import pulumi
import pulumi_aws as aws

config = pulumi.Config()
_allowed_cidr = config.get("allowedCidr") or "0.0.0.0/0"


@dataclass
class Network:
    vpc_id: pulumi.Output[str]
    subnet_ids: pulumi.Output[List[str]]
    rds_security_group_id: pulumi.Output[str]
    ecs_security_group_id: pulumi.Output[str]
    compute_security_group_id: pulumi.Output[str]


def create_network() -> Network:
    default_vpc = aws.ec2.get_vpc(default=True)
    default_subnets = aws.ec2.get_subnets(
        filters=[aws.ec2.GetSubnetsFilterArgs(name="vpc-id", values=[default_vpc.id])]
    )

    # RDS: Postgres, reachable from `_allowed_cidr` - so a developer's own
    # `PostgresMetadataStoreConfig` can point straight at it, the same way
    # it already points at the local docker-compose Postgres.
    rds_sg = aws.ec2.SecurityGroup(
        "bettmensch-ai-rds",
        vpc_id=default_vpc.id,
        description="Allows inbound Postgres to the bettmensch.ai RDS instance",
        ingress=[
            aws.ec2.SecurityGroupIngressArgs(
                protocol="tcp", from_port=5432, to_port=5432, cidr_blocks=[_allowed_cidr]
            )
        ],
        egress=[
            aws.ec2.SecurityGroupEgressArgs(
                protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"]
            )
        ],
        tags={"bettmensch-ai": "true"},
    )

    # ECS: the frontend, reachable from `_allowed_cidr` on its own port -
    # no ALB in this "basic" stack (see docs/design-decisions.md), so the
    # service's own public IP is what a developer browses directly.
    ecs_sg = aws.ec2.SecurityGroup(
        "bettmensch-ai-ecs-frontend",
        vpc_id=default_vpc.id,
        description="Allows inbound HTTP to the bettmensch.ai frontend ECS service",
        ingress=[
            aws.ec2.SecurityGroupIngressArgs(
                protocol="tcp", from_port=8080, to_port=8080, cidr_blocks=[_allowed_cidr]
            )
        ],
        egress=[
            aws.ec2.SecurityGroupEgressArgs(
                protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"]
            )
        ],
        tags={"bettmensch-ai": "true"},
    )

    # Batch (Fargate)/Lambda: outbound only - pulling the task-runtime image
    # from ECR, calling S3/RDS/CloudWatch. No inbound rule needed at all.
    compute_sg = aws.ec2.SecurityGroup(
        "bettmensch-ai-compute",
        vpc_id=default_vpc.id,
        description="Egress-only security group for Batch (Fargate) jobs",
        egress=[
            aws.ec2.SecurityGroupEgressArgs(
                protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"]
            )
        ],
        tags={"bettmensch-ai": "true"},
    )

    return Network(
        vpc_id=pulumi.Output.from_input(default_vpc.id),
        subnet_ids=pulumi.Output.from_input(default_subnets.ids),
        rds_security_group_id=rds_sg.id,
        ecs_security_group_id=ecs_sg.id,
        compute_security_group_id=compute_sg.id,
    )
