DOCKERTAG=bettmensch88/bettmensch.ai-lightning:3.11
BETTMENSCH_AI_VERSION=$(git rev-parse --short HEAD)

echo "Docker base tag: ${DOCKERTAG}"
echo "Commit hash: ${BETTMENSCH_AI_VERSION}"
echo "Docker tag: ${DOCKERTAG}-${BETTMENSCH_AI_VERSION}"

docker build -t ${DOCKERTAG}-latest -t ${DOCKERTAG}-${BETTMENSCH_AI_VERSION} --build-arg="BETTMENSCH_AI_VERSION=${BETTMENSCH_AI_VERSION}" . 

docker push ${DOCKERTAG}-${BETTMENSCH_AI_VERSION} 
docker push ${DOCKERTAG}-latest