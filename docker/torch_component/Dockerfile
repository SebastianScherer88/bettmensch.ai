from pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime as base

ARG BETTMENSCH_AI_VERSION

ENV NAME_OF_PACKAGE=bettmensch_ai
ENV SUBDIR_IN_REPO=sdk

RUN echo "git+https://github.com/SebastianScherer88/bettmensch.ai.git@${BETTMENSCH_AI_VERSION}#egg=${NAME_OF_PACKAGE}&subdirectory=${SUBDIR_IN_REPO}"

RUN apt update 
RUN apt-get install pip git -y
RUN pip install -e "git+https://github.com/SebastianScherer88/bettmensch.ai.git@${BETTMENSCH_AI_VERSION}#egg=${NAME_OF_PACKAGE}&subdirectory=${SUBDIR_IN_REPO}"

# set cwd back to root so the artifact setup done in the init container resolves correctly
WORKDIR / 