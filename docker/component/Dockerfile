from python:3.11 as base

ARG BETTMENSCH_AI_VERSION

ENV NAME_OF_PACKAGE=bettmensch_ai
ENV SUBDIR_IN_REPO=sdk

RUN echo "git+https://github.com/SebastianScherer88/bettmensch.ai.git@${BETTMENSCH_AI_VERSION}#egg=${NAME_OF_PACKAGE}&subdirectory=${SUBDIR_IN_REPO}"

RUN pip install -e "git+https://github.com/SebastianScherer88/bettmensch.ai.git#egg=${NAME_OF_PACKAGE}&subdirectory=${SUBDIR_IN_REPO}"