export DOCKERTAG=bettmensch88/bettmensch.ai:3.11
export BETTMENSCH_AI_VERSION=${git rev-parse --short HEAD}

docker build -t ${DOCKERTAG}-latest -t ${DOCKERTAG}-${BETTMENSCH_AI_VERSION} --build-arg="BETTMENSCH_AI_VERSION=${BETTMENSCH_AI_VERSION}" . 

docker push ${DOCKERTAG}-${BETTMENSCH_AI_VERSION} 
docker push ${DOCKERTAG}-latest