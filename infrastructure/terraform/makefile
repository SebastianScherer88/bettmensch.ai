## VARIABLES
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
	@echo "::endgroup::"

platform.connect:
	@echo "::group::Connecting to the platform..."
	make kubernetes.configure
	make kubernetes.connect
	@echo "::endgroup::"

platform.test:
	@echo "::group:: Running e2e platform tests"
	make sdk.test SUITE=k8s PYTEST_FLAGS="$(PYTEST_FLAGS)"
	@echo "::endgroup::"

platform.delete_test_pipelines:
	@echo "::group:: Removing all pipelines"
	make sdk.test SUITE=k8s PYTEST_FLAGS="-m delete_pipelines"
	@echo "::endgroup::"

platform.delete_test_flows:
	@echo "::group:: Removing all flows"
	make sdk.test SUITE=k8s PYTEST_FLAGS="-m delete_flows"
	@echo "::endgroup::"

platform.down:
	@echo "::group::Tearing down platform infrastructure..."
	# empty the argo workflow artifact repository s3 bucket, otherwise we
	# wont be able to remove it from AWS
	aws s3 rm s3://bettmensch-ai-artifact-repository --recursive
	terraform -chdir=$(DIR) destroy -auto-approve
	@echo "::endgroup::"