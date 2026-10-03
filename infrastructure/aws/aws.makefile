## infrastructure/aws convenience targets - see infrastructure/aws/README.md
## for prerequisites (pulumi login, real AWS credentials, `pulumi stack init
## dev`). Pushing the frontend image additionally needs Docker + the AWS
## CLI (`aws ecr get-login-password`).
##
## `frontend.build`/`frontend.push` (docker/frontend/makefile) are untouched
## - they publish to Docker Hub. `frontend.push.ecr` here reuses that same
## locally-built image but pushes it to the ECR repo `infrastructure/aws`
## provisions instead, since this stack's ECS service pulls from ECR.
##
## The task-runtime image push targets and the AWS-gated test target have
## been stashed along with the rest of the remote-compute layer - see
## stash/aws-remote-compute/README.md.

AWS_INFRA_DIR=infrastructure/aws

aws.up:
	@echo "::group::Provisioning the basic AWS stack (infrastructure/aws)"
	cd $(AWS_INFRA_DIR) && pulumi up
	@echo "::endgroup::"

aws.down:
	@echo "::group::Tearing down the basic AWS stack (infrastructure/aws)"
	cd $(AWS_INFRA_DIR) && pulumi destroy
	@echo "::endgroup::"

aws.outputs:
	cd $(AWS_INFRA_DIR) && pulumi stack output --json

frontend.push.ecr:
	@echo "::group::Pushing frontend image to ECR"
	$(eval ECR_REPO := $(shell cd $(AWS_INFRA_DIR) && pulumi stack output ecr_frontend_repo_url))
	$(eval ECR_HOST := $(word 1,$(subst /, ,$(ECR_REPO))))
	aws ecr get-login-password | docker login --username AWS --password-stdin $(ECR_HOST)
	docker tag bettmensch88/bettmensch.ai-frontend:local $(ECR_REPO):latest
	docker push $(ECR_REPO):latest
	@echo "::endgroup::"
