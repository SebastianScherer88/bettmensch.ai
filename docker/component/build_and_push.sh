export DOCKERTAG=bettmensch88/bettmensch.ai:3.11

docker build -t ${DOCKERTAG} . 

docker push ${DOCKERTAG}