ARG BASE_IMAGE=python
ARG BASE_IMAGE_VERSION=3:11

from ${BASE_IMAGE}:${BASE_IMAGE_VERSION} as base

ARG COMMIT
ARG PACKAGE_EXTRA=pipelines

ENV PIP_INSTALL_SPEC="git+https://github.com/SebastianScherer88/bettmensch.ai.git@${COMMIT}#egg=bettmensch_ai[${PACKAGE_EXTRA}]&subdirectory=sdk"

RUN apt update 
RUN apt-get install pip git -y

RUN echo "Installing bettmensch_ai with the following pip -e spec: ${PIP_INSTALL_SPEC}"

RUN pip install -e "${PIP_INSTALL_SPEC}"

# set cwd back to root so the artifact setup done in the init container resolves correctly
WORKDIR /