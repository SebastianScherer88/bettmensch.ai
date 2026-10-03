"""Entrypoint: wires up network -> storage/registry/iam -> frontend, and
exports everything `infrastructure/aws/README.md`'s env-var mapping needs.

AWS Batch/Lambda/Step Functions remote compute (and the AWS-gated test
fixtures that needed them) have been stashed pending a redesign of the
artifact/metadata Client abstractions - see
`stash/aws-remote-compute/README.md`. This program currently provisions
only what the artifact/metadata stores and the frontend need: an S3 bucket,
an RDS Postgres instance, and an ECS service running the frontend.

Run `pulumi up` from this directory (see README.md for prerequisites -
`pulumi stack init dev`, `pulumi config set aws:region ...`, `pulumi config
set --secret dbPassword ...`).
"""

import pulumi

from frontend import create_frontend
from iam import create_iam
from network import create_network
from registry import create_registry
from storage import create_storage

network = create_network()
storage = create_storage(network)
registry = create_registry()
iam = create_iam(storage)
cluster, service = create_frontend(network, storage, registry, iam)

# --- Outputs: feed infrastructure/aws/README.md's env var mapping ---------

pulumi.export("s3_bucket", storage.bucket_name)
pulumi.export("rds_dsn", storage.dsn)
pulumi.export("ecr_frontend_repo_url", registry.frontend_repo_url)
pulumi.export("ecs_cluster_name", cluster.name)
pulumi.export("ecs_service_name", service.name)
