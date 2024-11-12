ARG BASE_IMAGE=python
ARG BASE_IMAGE_VERSION=3:11

from ${BASE_IMAGE}:${BASE_IMAGE_VERSION} as base

ARG SDK_EXTRAS=pipelines

RUN apt update 
RUN apt-get install pip git -y

RUN echo "Installing bettmensch_ai sdk with pipelines extras"

COPY --from=sdk . ./sdk
RUN pip install ./sdk[${SDK_EXTRAS}]

# set cwd back to root so the artifact setup done in the init container resolves correctly
WORKDIR /