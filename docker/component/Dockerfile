from python:3.11 as base

ARG BETTMENSCH_AI_VERSION

ENV PACKAGE_NAME=bettmensch_ai
ENV PACKAGE_EXTRA=pipelines
ENV REPO_SUBDIRECTORY=sdk
ENV PIP_INSTALL_LOCATION="git+https://github.com/SebastianScherer88/bettmensch.ai.git@${BETTMENSCH_AI_VERSION}#egg=${PACKAGE_NAME}[${PACKAGE_EXTRA}]&subdirectory=${REPO_SUBDIRECTORY}"

RUN echo "${PIP_INSTALL_LOCATION}"

RUN pip install -e "${PIP_INSTALL_LOCATION}"

# set cwd back to root so the artifact setup done in the init container resolves correctly
WORKDIR /