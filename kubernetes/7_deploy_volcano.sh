# installation manifest from here: https://github.com/argoproj/argo-workflows/releases/download/v<<ARGO_WORKFLOWS_VERSION>>/quick-start-minimal.yaml
# remove the --auth-mode client lines from the argo-server deployment container args to allow hera workflow submission from laptop via port forward
kubectl apply -f ./manifests/volcano-installation.yaml

# configure artifact repository to use s3 bucket ``
kubectl apply -f ./manifests/volcano-queue.yaml

# Confirm that the 'default' has 4 cpus.
kubectl get queues -o yaml