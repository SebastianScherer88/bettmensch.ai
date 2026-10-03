"""Stashed from `infrastructure/aws/registry.py` - the two task-runtime ECR
repos (Batch, Lambda - see `docker/task-runtime/`'s two Dockerfiles, also
stashed). Re-add these resources (and the corresponding `Registry` dataclass
fields/outputs) to the active `registry.py`/`__main__.py` when remote
compute comes back.
"""

import pulumi_aws as aws


def _create_remote_compute_registry():
    batch_repo = aws.ecr.Repository(
        "bettmensch-ai-batch-task-runtime",
        image_tag_mutability="MUTABLE",
        force_delete=True,
        tags={"bettmensch-ai": "true"},
    )
    lambda_repo = aws.ecr.Repository(
        "bettmensch-ai-lambda-task-runtime",
        image_tag_mutability="MUTABLE",
        force_delete=True,
        tags={"bettmensch-ai": "true"},
    )

    return {
        "batch_task_runtime_repo_url": batch_repo.repository_url,
        "lambda_task_runtime_repo_url": lambda_repo.repository_url,
    }
