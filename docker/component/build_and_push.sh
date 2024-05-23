export DOCKERTAG=bettmensch88/bettmensch.ai:3.11
export BETTMENSCH_AI_VERSION="0d06b91"

docker build -t ${DOCKERTAG}-${BETTMENSCH_AI_VERSION} --build-arg="BETTMENSCH_AI_VERSION=${BETTMENSCH_AI_VERSION}" . 

docker push ${DOCKERTAG}