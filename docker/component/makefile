## VARIABLES
DOCKER_IMAGE_BASE=bettmensch88/bettmensch.ai
COMPONENT?=base
COMMIT?=$(shell git rev-parse --short HEAD)
LOCAL_TAG=local
COMMIT_TAG=3.11-$(COMMIT)
LATEST_TAG=3.11-latest

component.build:
	@echo "::group::Building $(COMPONENT) component docker image"
	@echo "::group:: Working dir: $(PWD)"
	@echo "::group:: Commit: $(COMMIT)"
	@echo "::group:: Bettmensch AI version: $(BETTMENSCH_AI_VERSION)"
	docker compose -f docker/component/docker-compose.yaml build $(COMPONENT) --build-arg="COMMIT=${COMMIT}"
	@echo "::endgroup::"

component.push:
	@echo "::group::Pushing $(COMPONENT) component docker image"
	docker tag ${DOCKER_IMAGE_BASE}-${COMPONENT}:${LOCAL_TAG} ${DOCKER_IMAGE_BASE}-${COMPONENT}:${COMMIT_TAG}
	docker tag ${DOCKER_IMAGE_BASE}-${COMPONENT}:${LOCAL_TAG} ${DOCKER_IMAGE_BASE}-${COMPONENT}:${LATEST_TAG}
	docker push ${DOCKER_IMAGE_BASE}-${COMPONENT}:${COMMIT_TAG}
	docker push ${DOCKER_IMAGE_BASE}-${COMPONENT}:${LATEST_TAG}
	@echo "::endgroup::"