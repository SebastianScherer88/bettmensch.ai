## VARIABLES
DIR?=infrastructure/terraform
ARTIFACT_REPOSITORY_S3_BUCKET=bettmensch-ai-artifact-repository
PYTEST_SUBPATH?=
PYTEST_FLAGS?=

platform.init:
	@echo "::group::Initializing platform infrastructure environment"
	pip install boto3 # needed for python bucket emptying utility in 
	# `platform.down` target
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
	make sdk.test SUITE=k8s PYTEST_FLAGS="$(PYTEST_FLAGS)" PYTEST_SUBPATH="$(PYTEST_SUBPATH)"
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
	# if it exists, empty the argo workflow artifact repository s3 bucket, 
	# otherwise we wont be able to remove it from AWS
	python3 -c "import boto3;buckets=boto3.client('s3').list_buckets()['Buckets'];exists=len([b for b in buckets if b['Name']=='$(ARTIFACT_REPOSITORY_S3_BUCKET)'])==1;_ = boto3.resource('s3').Bucket('$(ARTIFACT_REPOSITORY_S3_BUCKET)').objects.all().delete() if exists else None"
	
	# tear down infra stack
	terraform -chdir=$(DIR) destroy -auto-approve
	@echo "::endgroup::"