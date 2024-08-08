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