## Stashed from infrastructure/aws/aws.makefile - re-add alongside the
## active targets when remote compute comes back. Needs AWS_INFRA_DIR
## (still defined in the active file).

task-runtime.push.batch:
	@echo "::group::Building and pushing the Batch task-runtime image"
	$(eval ECR_REPO := $(shell cd $(AWS_INFRA_DIR) && pulumi stack output ecr_batch_task_runtime_repo_url))
	$(eval ECR_HOST := $(word 1,$(subst /, ,$(ECR_REPO))))
	docker build -f docker/task-runtime/Dockerfile.batch -t $(ECR_REPO):latest .
	aws ecr get-login-password | docker login --username AWS --password-stdin $(ECR_HOST)
	docker push $(ECR_REPO):latest
	@echo "::endgroup::"

task-runtime.push.lambda:
	@echo "::group::Building and pushing the Lambda task-runtime image"
	$(eval ECR_REPO := $(shell cd $(AWS_INFRA_DIR) && pulumi stack output ecr_lambda_task_runtime_repo_url))
	$(eval ECR_HOST := $(word 1,$(subst /, ,$(ECR_REPO))))
	docker build -f docker/task-runtime/Dockerfile.lambda -t $(ECR_REPO):latest .
	aws ecr get-login-password | docker login --username AWS --password-stdin $(ECR_HOST)
	docker push $(ECR_REPO):latest
	@echo "::endgroup::"

# Needs the BETTMENSCH_AI_AWS_TEST_* environment variables exported first -
# see infrastructure/aws/README.md's outputs table.
pipelines.test.aws:
	@echo "::group::Running AWS-gated functional tests"
	pytest tests/functional/pipelines -m aws
	@echo "::endgroup::"
