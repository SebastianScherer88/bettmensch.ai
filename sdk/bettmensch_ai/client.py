import argo_workflows
from hera.workflows import WorkflowsService


def client() -> WorkflowsService:

    workflow_service = WorkflowsService(
        host="https://127.0.0.1:2746", verify_ssl=False, namespace="argo"
    )

    return workflow_service


def argo_client():
    configuration = argo_workflows.Configuration(host="https://127.0.0.1:2746")
    configuration.verify_ssl = False
    api_client = argo_workflows.ApiClient(configuration)

    return api_client


client = client()
