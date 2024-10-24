## VARIABLES
REGION?=us-east-2
CLUSTER_NAME?=bettmensch-ai
DIR?=infrastructure/terraform

platform.init:
	@echo "::group::Initializing platform infrastructure environment"
	terraform -chdir=$(DIR) init
		@echo "::endgroup::"

platform.plan:
	@echo "::group::Planning platform infrastructure"
	terraform -chdir=$(DIR) plan

platform.up:
	@echo "::group::Bringing up platform infrastructure..."
	make platform.init DIR=$(DIR)
	make platform.plan DIR=$(DIR)
	terraform -chdir=$(DIR) apply -auto-approve
	aws eks --region $(REGION) update-kubeconfig --name $(CLUSTER_NAME)
	make kubernetes.connect
	@echo "::endgroup::"

platform.down:
		@echo "::group::Tearing down platform infrastructure..."
		# empty the argo workflow artifact repository s3 bucket, otherwise we
		# wont be able to remove it from AWS
		aws s3 rm s3://bettmensch-ai-artifact-repository --recursive
		terraform -chdir=$(DIR) destroy -auto-approve
		@echo "::endgroup::"

platform.test:
		@echo "::group::E2E testing platform..."
		make platform.up
		make sdk.test SUITE=k8s PYTEST_FLAGS=$(PYTEST_FLAGS)
		# make platform.down
		@echo "::endgroup::"

platform.delete_pipelines:
		@echo "::group:: Removing all pipelines"
		make sdk.test SUITE=k8s PYTEST_FLAGS="-m delete_pipelines"

platform.delete_flows:
		@echo "::group:: Removing all flows"
		make sdk.test SUITE=k8s PYTEST_FLAGS="-m delete_flows"