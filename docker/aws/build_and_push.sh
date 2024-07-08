DOCKERTAG=bettmensch88/aws:latest

echo "Docker tag: ${DOCKERTAG}"

docker build -t ${DOCKERTAG} . 

docker push ${DOCKERTAG}docker system