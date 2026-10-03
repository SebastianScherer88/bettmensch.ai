"""Storage: the S3 bucket `S3ArtifactStore` talks to, and the RDS Postgres
instance `PostgresMetadataStore` talks to - the genuine-AWS counterparts of
`pipelines.docker-compose.yaml`'s MinIO/Postgres containers.

Neither creates any schema/tables of its own: `S3ArtifactStore` never
assumes a particular bucket layout beyond the `key()` convention it already
builds in Python, and `PostgresMetadataStore.__init__` already creates its
own tables (`CREATE TABLE IF NOT EXISTS`, behind an advisory lock) the first
time it connects - matching how this stack's local dev counterpart needs no
migration step either.
"""

from dataclasses import dataclass

import pulumi
import pulumi_aws as aws

from network import Network

config = pulumi.Config()
_db_username = config.get("dbUsername") or "bettmensch_ai"
_db_password = config.require_secret("dbPassword")
_db_name = "bettmensch_ai_metadata"


@dataclass
class Storage:
    bucket_name: pulumi.Output[str]
    bucket_arn: pulumi.Output[str]
    db_endpoint: pulumi.Output[str]
    db_port: pulumi.Output[int]
    db_name: str
    db_username: str
    dsn: pulumi.Output[str]


def create_storage(network: Network) -> Storage:
    bucket = aws.s3.BucketV2("bettmensch-ai-artifacts", tags={"bettmensch-ai": "true"})

    # Basic, dev-grade: single-AZ, no read replica, no automated-backup
    # retention beyond RDS's own 1-day default - a production deployment
    # would want more of all three; out of scope for "basic" (see
    # docs/design-decisions.md).
    subnet_group = aws.rds.SubnetGroup(
        "bettmensch-ai-metadata",
        subnet_ids=network.subnet_ids,
        tags={"bettmensch-ai": "true"},
    )
    db_instance = aws.rds.Instance(
        "bettmensch-ai-metadata",
        engine="postgres",
        instance_class="db.t4g.micro",
        allocated_storage=20,
        db_name=_db_name,
        username=_db_username,
        password=_db_password,
        db_subnet_group_name=subnet_group.name,
        vpc_security_group_ids=[network.rds_security_group_id],
        publicly_accessible=True,
        skip_final_snapshot=True,
        tags={"bettmensch-ai": "true"},
    )

    dsn = pulumi.Output.all(_db_username, _db_password, db_instance.address, db_instance.port).apply(
        lambda args: f"postgresql://{args[0]}:{args[1]}@{args[2]}:{args[3]}/{_db_name}"
    )

    return Storage(
        bucket_name=bucket.bucket,
        bucket_arn=bucket.arn,
        db_endpoint=db_instance.address,
        db_port=db_instance.port,
        db_name=_db_name,
        db_username=_db_username,
        dsn=pulumi.Output.secret(dsn),
    )
