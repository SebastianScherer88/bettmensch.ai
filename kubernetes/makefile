## VARIABLES
ARGO_HEAD_POD=$(shell kubectl get pods --selector=app=argo-server -n argo -o custom-columns=POD:metadata.name --no-headers)
ARGO_PORT?=2746
MLFLOW_HEAD_POD=$(shell kubectl get pods --selector=app=mlflow-server -n mlflow -o custom-columns=POD:metadata.name --no-headers)
MLFLOW_PORT?=5000

kubernetes.connect.mlflow:
	@echo "::group::Port fowarding the Mlflow server on the K8s cluster"
	kubectl -n mlflow port-forward $(MLFLOW_HEAD_POD) --address 0.0.0.0 $(MLFLOW_PORT):$(MLFLOW_PORT) > mlflow-port-forward.log & # background process
	@echo "::endgroup::"

kubernetes.connect.argo:
	@echo "::group::Port forwarding the ArgoWorkflows server to the K8s cluster"
	kubectl -n argo port-forward $(ARGO_HEAD_POD) --address 0.0.0.0 $(ARGO_PORT):$(ARGO_PORT) > argo-port-forward.log & # background process
	@echo "::endgroup::"

kubernetes.connect:
	kubernetes.connect.mlflow
	kubernetes.connect.argo