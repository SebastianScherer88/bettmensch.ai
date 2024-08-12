import datetime
import json
import os

import argo_workflows
from argo_workflows.api.workflow_service_api import WorkflowServiceApi
from argo_workflows.api.workflow_template_service_api import (
    WorkflowTemplateServiceApi,
)
from hera.workflows import WorkflowsService

ARGO_WORKFLOW_MODELS_DIR = "./data_models/workflows/argo"
HERA_WORKFLOW_MODELS_DIR = "./data_models/workflows/hera"
ARGO_WORKFLOW_TEMPLATE_MODELS_DIR = "./data_models/workflow_templates/argo"
HERA_WORKFLOW_TEMPLATE_MODELS_DIR = "./data_models/workflow_templates/hera"
DIRS = [
    ARGO_WORKFLOW_MODELS_DIR,
    HERA_WORKFLOW_MODELS_DIR,
    ARGO_WORKFLOW_TEMPLATE_MODELS_DIR,
    HERA_WORKFLOW_TEMPLATE_MODELS_DIR,
]
for DIR in DIRS:
    if not os.path.exists(DIR):
        os.makedirs(DIR)


def recursive_datetime_to_string(data):
    if isinstance(data, dict):
        return dict(
            [(k, recursive_datetime_to_string(v)) for k, v in data.items()]
        )
    elif isinstance(data, list):
        return [recursive_datetime_to_string(data_i) for data_i in data]
    elif isinstance(data, tuple):
        return (recursive_datetime_to_string(data_i) for data_i in data)
    elif isinstance(data, datetime.datetime):
        return "test-datetime-value"
    else:
        return data


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
    argo_workflow_template_dict = recursive_datetime_to_string(
        argo_workflow_template.to_dict()
    )
    with open(
        f"{ARGO_WORKFLOW_TEMPLATE_MODELS_DIR}/argo_workflow_template_{i}.json",
        "w",
    ) as argo_workflow_template_file:
        json.dump(argo_workflow_template_dict, argo_workflow_template_file)

# export argo workflow
for i, argo_workflow in enumerate(argo_workflow_list_response.items):
    argo_workflow_dict = recursive_datetime_to_string(argo_workflow.to_dict())
    with open(
        f"{ARGO_WORKFLOW_MODELS_DIR}/argo_workflow_{i}.json", "w"
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
    hera_workflow_template_dict = recursive_datetime_to_string(
        hera_workflow_template.dict()
    )
    with open(
        f"{HERA_WORKFLOW_TEMPLATE_MODELS_DIR}/hera_workflow_template_{i}.json",
        "w",
    ) as hera_workflow_template_file:
        json.dump(hera_workflow_template_dict, hera_workflow_template_file)

# export hera workflow
for i, hera_workflow in enumerate(hera_workflow_list_response.items):
    hera_workflow_dict = recursive_datetime_to_string(hera_workflow.dict())
    with open(
        f"{HERA_WORKFLOW_MODELS_DIR}/hera_workflow_{i}.json", "w"
    ) as hera_workflow_file:
        json.dump(hera_workflow_dict, hera_workflow_file)
