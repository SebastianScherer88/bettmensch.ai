# `infrastructure/aws`: a basic, dev-grade AWS stack for `pipelines`

Pulumi (Python) program provisioning genuine AWS counterparts of
`pipelines.docker-compose.yaml`'s local dev stack:

- an **S3 bucket** for `S3ArtifactStore`
- an **RDS Postgres instance** for `PostgresMetadataStore`
- **IAM roles** for the frontend's ECS task
- an **ECR repository** for the frontend image
- an **ECS** Fargate cluster/service running the frontend

AWS Batch/Lambda/Step Functions remote compute (and the IAM roles, ECR
repos, Batch compute environment, and AWS-gated test fixtures that went
with it) have been stashed pending a redesign of the artifact/metadata
Client abstractions - see `stash/aws-remote-compute/README.md` for what
moved and how to bring it back.

Deliberately "basic," not production-hardened - see
`docs/design-decisions.md` for each trade-off (default VPC/no NAT, no ALB,
`0.0.0.0/0`-reachable by default) called out explicitly rather than
silently decided.

## Prerequisites

- The Pulumi CLI and a Pulumi backend you're logged into (`pulumi login` -
  a local file backend works fine: `pulumi login file://~`).
- Real AWS credentials with permission to create the resources above
  (`aws configure`, or any of boto3's normal credential sources).
- Python 3.11+ (Pulumi's Python runtime).

## Provisioning

```bash
cd infrastructure/aws
python -m venv venv && source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

pulumi stack init dev
cp Pulumi.dev.yaml.example Pulumi.dev.yaml   # edit region/username/CIDR as needed
pulumi config set --secret dbPassword <a-real-password>

pulumi up
```

`pulumi up` will succeed for everything except the frontend ECS service
reaching a healthy state, until you've pushed a real image - see below.
That's expected: the ECR repo has to exist before you can push to it, and
this stack creates the repo and everything else in the same `pulumi up`.

## Pushing the frontend image

From the repo root, after `pulumi up` has created the ECR repo:

```bash
make aws.outputs        # prints every output below as JSON

make frontend.build     # existing Docker Hub build target
make frontend.push.ecr  # pushes that same image to the ECR repo above
```

Then `pulumi up` again so the frontend service picks up the now-real image.

## Outputs -> environment variables

`pulumi stack output --json` (or `make aws.outputs`) prints every value
below. Point a real `S3ArtifactStoreConfig`/`PostgresMetadataStoreConfig`
at this stack (mirroring how `pipelines.docker-compose.yaml`'s forwarded
`localhost` ports work for local dev) via:

| Pulumi output | Environment variable |
| --- | --- |
| `s3_bucket` | `BETTMENSCH_AI_S3_ARTIFACT_STORE_BUCKET` |
| `rds_dsn` | `BETTMENSCH_AI_POSTGRES_METADATA_STORE_DSN` |

## Tearing down

```bash
cd infrastructure/aws
pulumi destroy
```
