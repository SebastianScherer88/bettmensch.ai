## VARIABLES
REGION?=us-east-2
CLUSTER_NAME?=bettmensch-ai
DIR?=infrastructure/terraform

infrastructure.build:
	@echo "::group::Building infrastructure"
	terraform -chdir=$(DIR) init
	terraform -chdir=$(DIR) plan
	terraform -chdir=$(DIR) apply -auto-approve
	aws eks --region $(REGION) update-kubeconfig --name $(CLUSTER_NAME)
	@echo "::endgroup::"

infrastructure.destroy:
		@echo "::group::Tearing down infrastructure"
		# empty the argo workflow artifact repository s3 bucket, otherwise we
		# wont be able to remove it from AWS
		aws s3 rm s3://bettmensch-ai-artifact-repository --recursive
		terraform -chdir=$(DIR) destroy -auto-approve
		@echo "::endgroup::"