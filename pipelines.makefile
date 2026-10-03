## pipelines test/install commands.
##
## Uses `uv` + the root pyproject.toml as the one and only packaging/
## dependency mechanism for this project. It's also the root of a small uv
## workspace: docker/frontend and docker/metadata-service each have their
## own pyproject.toml for their own deployment-only dependencies
## (fastapi/uvicorn, psycopg where needed) - see their Dockerfiles. The
## root `dev` dependency group covers everything needed to run the *test*
## suite locally (pytest, psycopg, fastapi, uvicorn), so a bare `uv sync`
## is enough; the root `postgres` extra remains separately for a
## production consumer of the `PostgresMetadataStore` abstraction itself.
##
## SCOPE variables:
##   SUITE      unit | integration | functional | all (default: unit)
##   MODULE     optional, narrows SUITE to one file, e.g.
##              test_metadata_store.py (only meaningful when SUITE isn't
##              "all" - which suite a bare module name belongs to is
##              otherwise ambiguous)
##   TEST_CASE  optional, narrows further to one test function/class within
##              MODULE, e.g. test_start_pipeline_run_records_a_running_status
##   PYTEST_FLAGS  passed straight through to pytest, e.g. "-k foo", "-x"
##
## Examples:
##   make pipelines.test
##   make pipelines.test SUITE=all
##   make pipelines.test SUITE=integration MODULE=test_s3_artifact_store_integration.py
##   make pipelines.test SUITE=unit MODULE=test_metadata_store.py TEST_CASE=test_list_triggers_is_empty_when_none_registered
##   make pipelines.test.docker SUITE=functional

## VARIABLES
COMPOSE_FILE=docker-compose/pipelines.docker-compose.yaml
SUITE?=unit
MODULE?=
TEST_CASE?=
PYTEST_FLAGS?=

ifeq ($(SUITE),all)
PIPELINES_TEST_TARGET=tests/unit/pipelines tests/integration/pipelines tests/functional/pipelines
else
PIPELINES_TEST_TARGET=tests/$(SUITE)/pipelines
ifneq ($(MODULE),)
PIPELINES_TEST_TARGET:=$(PIPELINES_TEST_TARGET)/$(MODULE)
endif
ifneq ($(TEST_CASE),)
PIPELINES_TEST_TARGET:=$(PIPELINES_TEST_TARGET)::$(TEST_CASE)
endif
endif

## INSTALLATION

pipelines.install:
	@echo "::group::Installing pipelines dependencies via uv (dev group: covers the full test suite)"
	uv sync
	@echo "::endgroup::"

## TEST INFRASTRUCTURE (postgres, minio - see pipelines.docker-compose.yaml)

pipelines.docker.up:
	@echo "::group::Starting pipelines test infrastructure (postgres, minio)"
	docker compose -f $(COMPOSE_FILE) up -d postgres minio
	@echo "::endgroup::"

pipelines.docker.down:
	@echo "::group::Stopping pipelines test infrastructure"
	docker compose -f $(COMPOSE_FILE) down -v
	@echo "::endgroup::"

## LOCAL DEV STACK (postgres, minio, and the read-only frontend viewer -
## the full "usage (3)" stack from pipelines.docker-compose.yaml's own
## header comment. Point your own LocalRunner scripts' S3ArtifactStore/
## PostgresMetadataStore at the forwarded ports and browse the results at
## http://localhost:8501.)

pipelines.up:
	@echo "::group::Starting pipelines local dev stack (postgres, minio, frontend)"
	docker compose -f $(COMPOSE_FILE) up -d --build postgres minio createbuckets frontend
	@echo "Frontend: http://localhost:8501"
	@echo "::endgroup::"

pipelines.down:
	@echo "::group::Stopping pipelines local dev stack"
	docker compose -f $(COMPOSE_FILE) down -v
	@echo "::endgroup::"

## TEST RUNNER IMAGE (the `test` service - a containerized pytest,
## in-network with postgres/minio; see the Dockerfile alongside the
## compose file)

pipelines.docker.build:
	@echo "::group::Building pipelines test runner image"
	docker compose -f $(COMPOSE_FILE) --profile test build test
	@echo "::endgroup::"

## TEST INVOCATION

# Runs from the host, against postgres/minio's forwarded ports (or
# BETTMENSCH_AI_TEST_* env vars pointed elsewhere) - `SUITE=unit` needs
# neither `pipelines.docker.up` nor any real infrastructure at all;
# `integration`/`functional`/`all` need `pipelines.docker.up` first, or
# their infra-dependent tests just skip.
pipelines.test:
	@echo "::group::Running pipelines tests ($(SUITE)): $(PIPELINES_TEST_TARGET)"
	pytest $(PIPELINES_TEST_TARGET) $(PYTEST_FLAGS)
	@echo "::endgroup::"

# Fully containerized: brings up postgres/minio, builds the test runner
# image, and runs pytest *inside* the compose network against them by
# service name - closer to how CI would actually see this than (1) above.
pipelines.test.docker: pipelines.docker.up pipelines.docker.build
	@echo "::group::Running pipelines tests in docker ($(SUITE)): $(PIPELINES_TEST_TARGET)"
	docker compose -f $(COMPOSE_FILE) --profile test run --rm test \
		pytest $(PIPELINES_TEST_TARGET) $(PYTEST_FLAGS)
	@echo "::endgroup::"
