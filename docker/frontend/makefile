## VARIABLES
DOCKER_ACCOUNT=bettmensch88
DOCKER_REPO_BASE=bettmensch.ai
DOCKER_IMAGE_BASE=${DOCKER_ACCOUNT}/$(DOCKER_REPO_BASE)
LOG_DIR?=build_logs
COMMIT?=$(shell git rev-parse --short HEAD)
LOCAL_TAG=local
COMMIT_TAG=3.11-$(COMMIT)
LATEST_TAG=3.11-latest

frontend.build:
	@echo "::group::Building frontend docker image"
	@echo "::group:: Commit: $(COMMIT)"
	mkdir -p ${LOG_DIR}
	docker compose -f docker/frontend/docker-compose.yaml build frontend > ${LOG_DIR}/docker-build-log-frontend.txt
	docker tag $(DOCKER_IMAGE_BASE)-frontend:${LOCAL_TAG} ${DOCKER_IMAGE_BASE}-frontend:${COMMIT_TAG}
	docker tag $(DOCKER_IMAGE_BASE)-frontend:${LOCAL_TAG} ${DOCKER_IMAGE_BASE}-frontend:${LATEST_TAG}
	@echo "::endgroup::"

frontend.push:
	@echo "::group::Pushing frontend docker image"
	docker push ${DOCKER_IMAGE_BASE}-frontend:${COMMIT_TAG}
	docker push ${DOCKER_IMAGE_BASE}-frontend:${LATEST_TAG}
	@echo "::endgroup::"

# For local dev/testing, prefer `make pipelines.up` instead (also brings up
# postgres/minio) - this is for running the *published* image standalone.
frontend.run:
	@echo "::group::Running frontend docker image locally"
	docker run --rm -p 8080:8080 \
		-e BETTMENSCH_AI_POSTGRES_METADATA_STORE_DSN \
		-e BETTMENSCH_AI_S3_ARTIFACT_STORE_BUCKET \
		-e BETTMENSCH_AI_S3_ARTIFACT_STORE_ENDPOINT_URL \
		-e BETTMENSCH_AI_S3_ARTIFACT_STORE_AWS_ACCESS_KEY_ID \
		-e BETTMENSCH_AI_S3_ARTIFACT_STORE_AWS_SECRET_ACCESS_KEY \
		$(DOCKER_IMAGE_BASE)-frontend:${LOCAL_TAG}
	@echo "::endgroup::"
