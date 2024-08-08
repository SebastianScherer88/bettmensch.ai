ARG BASE_IMAGE=python
ARG BASE_IMAGE_VERSION=3:11

from ${BASE_IMAGE}:${BASE_IMAGE_VERSION} as base

WORKDIR /app

RUN pwd

RUN echo "Installing bettmensch_ai sdk with dashboard extras"

RUN apt-get update && apt-get install libgl1 -y # needed for cv2 in docker

COPY --from=sdk . ./sdk
RUN pip install ./sdk[dashboard]

COPY --from=image . ./image
COPY --from=config . ./.streamlit
COPY --from=src . ./src

EXPOSE 8501