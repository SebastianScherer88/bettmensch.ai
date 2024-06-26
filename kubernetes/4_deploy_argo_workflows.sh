kubectl create namespace argo

# installation manifest from here: https://github.com/argoproj/argo-workflows/releases/download/v<<ARGO_WORKFLOWS_VERSION>>/quick-start-minimal.yaml
# remove the --auth-mode client lines from the argo-server deployment container args to allow hera workflow submission from laptop via port forward
kubectl apply -n argo -f ./manifests/argo/argo-workflow-installation.yaml

# configure artifact repository to use s3 bucket ``
kubectl apply -n argo -f ./manifests/argo/argo-artifact-repository-config.yaml

# Confirm that the operator is running in the namespace `default`.
kubectl get pods -n argo -w