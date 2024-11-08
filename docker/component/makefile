## VARIABLES
DOCKER_ACCOUNT=bettmensch88
DOCKER_REPO_BASE=bettmensch.ai
DOCKER_IMAGE_BASE=${DOCKER_ACCOUNT}/$(DOCKER_REPO_BASE)
COMPONENT?=standard
COMMIT?=$(shell git rev-parse --short HEAD)
LOCAL_TAG=local
COMMIT_TAG=3.11-$(COMMIT)
LATEST_TAG=3.11-latest

component.build:
	@echo "::group::Building $(COMPONENT) component docker image"
	@echo "::group:: Commit: $(COMMIT)"
	docker compose -f docker/component/docker-compose.yaml build $(COMPONENT) > docker-build-log-$(COMPONENT).txt
	docker tag $(DOCKER_IMAGE_BASE)-${COMPONENT}:${LOCAL_TAG} ${DOCKER_IMAGE_BASE}-${COMPONENT}:${COMMIT_TAG}
	docker tag ${DOCKER_IMAGE_BASE}-${COMPONENT}:${LOCAL_TAG} ${DOCKER_IMAGE_BASE}-${COMPONENT}:${LATEST_TAG}
	@echo "::endgroup::"

component.push:
	@echo "::group::Pushing $(COMPONENT) component docker image"
	docker push ${DOCKER_IMAGE_BASE}-${COMPONENT}:${COMMIT_TAG}
	docker push ${DOCKER_IMAGE_BASE}-${COMPONENT}:${LATEST_TAG}
	@echo "::endgroup::"