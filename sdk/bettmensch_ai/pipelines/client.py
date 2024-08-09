from hera.workflows import WorkflowsService


def client() -> WorkflowsService:

    workflow_service = WorkflowsService(
        host="https://127.0.0.1:2746", verify_ssl=False, namespace="argo"
    )

    return workflow_service


hera_client = client()
