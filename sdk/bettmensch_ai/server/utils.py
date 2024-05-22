from typing import List

import argo_workflows
from argo_workflows.api import (workflow_service_api,
                                workflow_template_service_api)
from argo_workflows.model.io_argoproj_workflow_v1alpha1_workflow import \
    IoArgoprojWorkflowV1alpha1Workflow
from argo_workflows.model.io_argoproj_workflow_v1alpha1_workflow_template import \
    IoArgoprojWorkflowV1alpha1WorkflowTemplate

PIPELINE_NODE_EMOJI_MAP = {
    "task": "🔵",  # :large_blue_circle:
    "inputs": {
        "task": "⤵️",  # :arrow_heading_down:
        "pipeline": "⏬",  # :arrow_double_down:
    },
    "outputs": {"task": "↪️"},  # :arrow_right_hook:
}

# --- ArgoWorkflow server config
def configure_argo_server():
    # get a sample pipeline from the ArgoWorkflow server
    configuration = argo_workflows.Configuration(host="https://127.0.0.1:2746")
    configuration.verify_ssl = False

    return configuration


def get_workflow_templates(
    configuration,
) -> List[IoArgoprojWorkflowV1alpha1WorkflowTemplate]:
    api_client = argo_workflows.ApiClient(configuration)
    api_instance = workflow_template_service_api.WorkflowTemplateServiceApi(api_client)

    return api_instance.list_workflow_templates(namespace="argo")["items"]


def get_workflows(configuration) -> List[IoArgoprojWorkflowV1alpha1Workflow]:
    api_client = argo_workflows.ApiClient(configuration)
    api_instance = workflow_service_api.WorkflowServiceApi(api_client)

    return api_instance.list_workflows(namespace="argo")["items"]


configuration = configure_argo_server()
