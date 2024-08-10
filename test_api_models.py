import json
import os

import argo_workflows
from argo_workflows.api.workflow_service_api import WorkflowServiceApi
from argo_workflows.api.workflow_template_service_api import (
    WorkflowTemplateServiceApi,
)
from hera.workflows import WorkflowsService

ARGO_DATA_MODELS_DIR = "./data_models/argo"
HERA_DATA_MODELS_DIR = "./data_models/hera"

if not os.path.exists(ARGO_DATA_MODELS_DIR):
    os.makedirs(ARGO_DATA_MODELS_DIR)

if not os.path.exists(HERA_DATA_MODELS_DIR):
    os.makedirs(HERA_DATA_MODELS_DIR)

# --- argo
# argo client
argo_configuration = argo_workflows.Configuration(
    host="https://127.0.0.1:2746"
)  # noqa: E501
argo_configuration.verify_ssl = False

argo_api_client = argo_workflows.ApiClient(argo_configuration)
argo_workflow_template_api = WorkflowTemplateServiceApi(argo_api_client)
argo_workflow_api = WorkflowServiceApi(argo_api_client)

argo_workflow_template_list_response = (
    argo_workflow_template_api.list_workflow_templates(namespace="argo")
)
argo_workflow_list_response = argo_workflow_api.list_workflows(
    namespace="argo"
)  # noqa: E501

# export argo workflow templates
for i, argo_workflow_template in enumerate(
    argo_workflow_template_list_response.items
):
    argo_workflow_template_dict = argo_workflow_template.to_dict()
    argo_workflow_template_dict["metadata"]["creation_timestamp"] = "test-time"
    argo_workflow_template_dict["metadata"]["managed_fields"][0][
        "time"
    ] = "test-time"
    with open(
        f"{ARGO_DATA_MODELS_DIR}/argo_workflow_template_{i}.json", "w"
    ) as argo_workflow_template_file:
        json.dump(argo_workflow_template_dict, argo_workflow_template_file)

# export argo workflow
for i, argo_workflow in enumerate(argo_workflow_list_response.items):
    argo_workflow_dict = argo_workflow.to_dict()
    argo_workflow_dict["metadata"]["creation_timestamp"] = "test-time"
    argo_workflow_dict["metadata"]["managed_fields"][0]["time"] = "test-time"
    with open(
        f"{ARGO_DATA_MODELS_DIR}/argo_workflow_{i}.json", "w"
    ) as argo_workflow_file:
        json.dump(argo_workflow_dict, argo_workflow_file)

# --- hera
# hera client

hera_api = WorkflowsService(
    host="https://127.0.0.1:2746", verify_ssl=False, namespace="argo"
)

hera_workflow_template_list_response = hera_api.list_workflow_templates()
hera_workflow_list_response = hera_api.list_workflows()

# export hera workflow templates
for i, hera_workflow_template in enumerate(
    hera_workflow_template_list_response.items
):
    hera_workflow_template_dict = hera_workflow_template.dict()
    hera_workflow_template_dict["metadata"]["creation_timestamp"] = "test-time"
    hera_workflow_template_dict["metadata"]["managed_fields"][0][
        "time"
    ] = "test-time"
    with open(
        f"{HERA_DATA_MODELS_DIR}/hera_workflow_template_{i}.json", "w"
    ) as hera_workflow_template_file:
        json.dump(hera_workflow_template_dict, hera_workflow_template_file)

# export hera workflow
for i, hera_workflow in enumerate(hera_workflow_list_response.items):
    hera_workflow_dict = hera_workflow.dict()
    hera_workflow_dict["metadata"]["creation_timestamp"] = "test-time"
    hera_workflow_dict["metadata"]["managed_fields"][0]["time"] = "test-time"
    with open(
        f"{HERA_DATA_MODELS_DIR}/hera_workflow_template_{i}.json", "w"
    ) as hera_workflow_file:
        json.dump(hera_workflow_dict, hera_workflow_file)
