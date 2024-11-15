from hera.workflows import WorkflowsService
from pydantic_settings import BaseSettings, SettingsConfigDict


# --- ArgoWorkflow server config
class ArgoWorkflowsBackendConfiguration(BaseSettings):
    host: str = "https://127.0.0.1:2746"
    verify_ssl: bool = False
    namespace: str = "argo"

    model_config = SettingsConfigDict(env_prefix="argo_workflows_backend_")


hera_workflow_service_configuration = ArgoWorkflowsBackendConfiguration()

hera_client = WorkflowsService(
    host=hera_workflow_service_configuration.host,
    verify_ssl=hera_workflow_service_configuration.verify_ssl,
    namespace=hera_workflow_service_configuration.namespace,
)
