"""ECR: the frontend's own image repo for the ECS service.

The two task-runtime repos (Batch, Lambda) have been stashed along with the
rest of that remote-compute layer - see `stash/aws-remote-compute/README.md`.
Lambda container images can only be pulled from ECR in the same account/
region regardless of preference, so the frontend uses ECR here too rather
than mixing in the existing Docker Hub publishing path (`docker/frontend/
makefile`'s `frontend.push`), which stays untouched for whoever still wants
a standalone Docker Hub image.
"""

from dataclasses import dataclass

import pulumi_aws as aws


@dataclass
class Registry:
    frontend_repo_url: str


def create_registry() -> Registry:
    frontend_repo = aws.ecr.Repository(
        "bettmensch-ai-frontend",
        image_tag_mutability="MUTABLE",
        force_delete=True,
        tags={"bettmensch-ai": "true"},
    )

    return Registry(frontend_repo_url=frontend_repo.repository_url)
