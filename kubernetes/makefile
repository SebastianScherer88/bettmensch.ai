## VARIABLES
REGION?=us-east-2
CLUSTER_NAME?=bettmensch-ai
ARGO_SERVICE?=argo-server
ARGO_NAMESPACE?=argo
ARGO_PORT?=2746
MLFLOW_SERVICE?=mlflow-server
MLFLOW_NAMESPACE=mlflow
MLFLOW_PORT?=5000
DASHBOARD_SERVICE?=bettmensch-ai-dashboard
DASHBOARD_NAMESPACE?=bettmensch-ai
DASHBOARD_PORT=8501

kubernetes.configure:
	@echo "::group::Configuring kubeconfig"
	aws eks --region $(REGION) update-kubeconfig --name $(CLUSTER_NAME)
	@echo "::endgroup::"

kubernetes.connect_service:
	@echo "::group::Port fowarding the Mlflow server on the K8s cluster"
	kubectl -n $(NAMESPACE) port-forward service/$(SERVICE) --address 0.0.0.0 $(PORT):$(PORT) > port-forward-$(SERVICE).log & # background process
	@echo "::endgroup::"

kubernetes.connect.argo:
	@echo "::group::Port fowarding the ArgoWorkflows server on the K8s cluster"
	make kubernetes.connect_service NAMESPACE=$(ARGO_NAMESPACE) SERVICE=$(ARGO_SERVICE) PORT=$(ARGO_PORT)
	@echo "::endgroup::"

kubernetes.connect.mlflow:
	@echo "::group::Port fowarding the Mlflow server on the K8s cluster"
	make kubernetes.connect_service NAMESPACE=$(MLFLOW_NAMESPACE) SERVICE=$(MLFLOW_SERVICE) PORT=$(MLFLOW_PORT)
	@echo "::endgroup::"

kubernetes.connect.dashboard:
	@echo "::group::Port fowarding the Bettmensch.ai dashboard on the K8s cluster"
	make kubernetes.connect_service NAMESPACE=$(DASHBOARD_NAMESPACE) SERVICE=$(DASHBOARD_SERVICE) PORT=$(DASHBOARD_PORT)
	@echo "::endgroup::"

kubernetes.connect:
	# make kubernetes.connect.mlflow
	make kubernetes.connect.argo
	make kubernetes.connect.dashboard