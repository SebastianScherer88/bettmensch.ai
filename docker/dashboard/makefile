## VARIABLES
DOCKER_ACCOUNT=bettmensch88
DOCKER_REPO_BASE=bettmensch.ai
DOCKER_IMAGE_BASE=${DOCKER_ACCOUNT}/$(DOCKER_REPO_BASE)
DASHBOARD=dashboard
LOG_DIR?=build_logs
COMMIT?=$(shell git rev-parse --short HEAD)
LOCAL_TAG=local
COMMIT_TAG=3.11-$(COMMIT)
LATEST_TAG=3.11-latest

dashboard.build:
	@echo "::group::Building $(DASHBOARD) docker image"
	@echo "::group:: Commit: $(COMMIT)"
	docker compose -f docker/$(DASHBOARD)/docker-compose.yaml build $(DASHBOARD) > ${LOG_DIR}/docker-build-log-${COMPONENT}.txt
	docker tag ${DOCKER_IMAGE_BASE}-${DASHBOARD}:${LOCAL_TAG} ${DOCKER_IMAGE_BASE}-${DASHBOARD}:${COMMIT_TAG}
	docker tag ${DOCKER_IMAGE_BASE}-${DASHBOARD}:${LOCAL_TAG} ${DOCKER_IMAGE_BASE}-${DASHBOARD}:${LATEST_TAG}
	@echo "::endgroup::"

dashboard.run:
	@echo "::group::Running $(DASHBOARD) docker image"
	docker compose -f docker/$(DASHBOARD)/docker-compose.yaml run $(DASHBOARD)
	@echo "::endgroup::"

dashboard.push:
	@echo "::group::Pushing $(DASHBOARD) docker image"
	docker push ${DOCKER_IMAGE_BASE}-${DASHBOARD}:${COMMIT_TAG}
	docker push ${DOCKER_IMAGE_BASE}-${DASHBOARD}:${LATEST_TAG}
	@echo "::endgroup::"