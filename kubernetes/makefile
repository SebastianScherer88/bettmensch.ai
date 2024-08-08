## VARIABLES
HEAD_POD=$(shell kubectl get pods --selector=app=argo-server -n argo -o custom-columns=POD:metadata.name --no-headers)
PORT?=2746

kubernetes.connect:
	@echo "::group::Connecting to the K8s cluster"
	kubectl -n argo port-forward $(HEAD_POD) --address 0.0.0.0 $(PORT):$(PORT)
	@echo "::endgroup::"