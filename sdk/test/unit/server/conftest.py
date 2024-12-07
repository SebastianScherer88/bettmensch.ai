import os
from datetime import datetime

import pytest
from hera.workflows.models import Workflow as WorkflowModel
from hera.workflows.models import WorkflowTemplate as WorkflowTemplateModel


@pytest.fixture
def test_output_dir():
    return os.path.join(".", "sdk", "test", "unit", "outputs")


@pytest.fixture
def test_datetime():
    return datetime(2024, 12, 1)


@pytest.fixture
def test_hera_artifact_workflow_template_model(test_datetime):

    return WorkflowTemplateModel(
        metadata={
            "creation_timestamp": test_datetime,
            "generate_name": "pipeline-test-artifact-pipeline-",
            "generation": 1,
            "labels": {
                "workflows.argoproj.io/creator": "system-serviceaccount-argo-argo-server"
            },
            "managed_fields": [
                {
                    "api_version": "argoproj.io/v1alpha1",
                    "fields_type": "FieldsV1",
                    "fields_v1": {},
                    "manager": "argo",
                    "operation": "Update",
                    "time": test_datetime,
                }
            ],
            "name": "pipeline-test-artifact-pipeline-jx7pb",
            "namespace": "argo",
            "resource_version": "7515",
            "uid": "e2e6b22b-4dfc-413d-ad43-f06a3b03cb92",
        },
        spec={
            "arguments": {"parameters": [{"name": "a", "value": "Param A"}]},
            "entrypoint": "bettmensch-ai-outer-dag",
            "templates": [
                {
                    "dag": {
                        "tasks": [
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "a",
                                            "value": "{{inputs.parameters.a}}",
                                        }
                                    ]
                                },
                                "name": "convert-to-artifact-0",
                                "template": "convert-to-artifact",
                            },
                            {
                                "arguments": {
                                    "artifacts": [
                                        {
                                            "from_": "{{tasks.convert-to-artifact-0.outputs.artifacts.a_art}}",
                                            "name": "a",
                                        }
                                    ]
                                },
                                "depends": "convert-to-artifact-0",
                                "name": "show-artifact-0",
                                "template": "show-artifact",
                            },
                        ]
                    },
                    "inputs": {
                        "parameters": [{"name": "a", "value": "Param A"}]
                    },
                    "metadata": {},
                    "name": "bettmensch-ai-inner-dag",
                    "outputs": {
                        "artifacts": [
                            {
                                "from_": "{{tasks.show-artifact-0.outputs.artifacts.b}}",
                                "name": "b",
                            }
                        ]
                    },
                },
                {
                    "inputs": {
                        "parameters": [
                            {"name": "a"},
                            {"default": "null", "name": "a_art"},
                        ]
                    },
                    "metadata": {},
                    "name": "convert-to-artifact",
                    "outputs": {
                        "artifacts": [{"name": "a_art", "path": "a_art"}]
                    },
                    "retry_strategy": {
                        "limit": "1",
                        "retry_policy": "OnError",
                    },
                    "script": {
                        "command": ["python"],
                        "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                        "image_pull_policy": "Always",
                        "name": "",
                        "resources": {
                            "limits": {"cpu": "100m", "memory": "100Mi"},
                            "requests": {"cpu": "100m", "memory": "100Mi"},
                        },
                        "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: a = json.loads(r'''{{inputs.parameters.a}}''')\nexcept: a = r'''{{inputs.parameters.a}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputArtifact\na_art = OutputArtifact(\"a_art\")\n\ndef convert_to_artifact(a: InputParameter, a_art: OutputArtifact=None) -> None:\n    \"\"\"When decorated with the bettmensch_ai.components.component decorator,\n    implements a bettmensch_ai.Component that converts its InputParameter into\n    an OutputArtifact.\"\"\"\n    with open(a_art.path, 'w') as a_art_file:\n        a_art_file.write(str(a))\n\nconvert_to_artifact(a,a_art)\n",
                    },
                },
                {
                    "inputs": {
                        "artifacts": [{"name": "a", "path": "a"}],
                        "parameters": [{"default": "null", "name": "b"}],
                    },
                    "metadata": {},
                    "name": "show-artifact",
                    "outputs": {"artifacts": [{"name": "b", "path": "b"}]},
                    "retry_strategy": {
                        "limit": "1",
                        "retry_policy": "OnError",
                    },
                    "script": {
                        "command": ["python"],
                        "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                        "image_pull_policy": "Always",
                        "name": "",
                        "resources": {
                            "limits": {"cpu": "100m", "memory": "100Mi"},
                            "requests": {"cpu": "100m", "memory": "100Mi"},
                        },
                        "source": 'import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import InputArtifact\na = InputArtifact("a")\n\nfrom bettmensch_ai.pipelines.io import OutputArtifact\nb = OutputArtifact("b")\n\ndef show_artifact(a: InputArtifact, b: OutputArtifact=None) -> None:\n    """When decorated with the bettmensch_ai.components.component decorator,\n    implements a bettmensch_ai.Component that prints the values of its\n    InputArtifact."""\n    with open(a.path, \'r\') as a_art_file:\n        a_content = a_art_file.read()\n    print(f\'Content of input artifact a: {a_content}\')\n    with open(b.path, \'w\') as b_art_file:\n        b_art_file.write(str(a_content))\n\nshow_artifact(a,b)\n',
                    },
                },
                {
                    "dag": {
                        "tasks": [
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "a",
                                            "value": "{{workflow.parameters.a}}",
                                        }
                                    ]
                                },
                                "name": "bettmensch-ai-inner-dag",
                                "template": "bettmensch-ai-inner-dag",
                            }
                        ]
                    },
                    "inputs": {},
                    "metadata": {},
                    "name": "bettmensch-ai-outer-dag",
                    "outputs": {},
                },
            ],
        },
    )


@pytest.fixture
def test_hera_parameter_workflow_template_model(test_datetime):

    return WorkflowTemplateModel(
        metadata={
            "creation_timestamp": test_datetime,
            "generate_name": "pipeline-test-parameter-pipeline-",
            "generation": 1,
            "labels": {
                "workflows.argoproj.io/creator": "system-serviceaccount-argo-argo-server"
            },
            "managed_fields": [
                {
                    "api_version": "argoproj.io/v1alpha1",
                    "fields_type": "FieldsV1",
                    "fields_v1": {},
                    "manager": "argo",
                    "operation": "Update",
                    "time": test_datetime,
                }
            ],
            "name": "pipeline-test-parameter-pipeline-c877j",
            "namespace": "argo",
            "resource_version": "7640",
            "uid": "d2715290-865d-4776-84c4-776632cd7159",
        },
        spec={
            "arguments": {
                "parameters": [
                    {"name": "a", "value": "1"},
                    {"name": "b", "value": "2"},
                ]
            },
            "entrypoint": "bettmensch-ai-outer-dag",
            "templates": [
                {
                    "dag": {
                        "tasks": [
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "a",
                                            "value": "{{inputs.parameters.a}}",
                                        },
                                        {
                                            "name": "b",
                                            "value": "{{inputs.parameters.b}}",
                                        },
                                    ]
                                },
                                "name": "a-plus-b-0",
                                "template": "a-plus-b",
                            },
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "a",
                                            "value": "{{tasks.a-plus-b-0.outputs.parameters.sum}}",
                                        },
                                        {"name": "b", "value": "2"},
                                    ]
                                },
                                "depends": "a-plus-b-0",
                                "name": "a-plus-b-plus-2-0",
                                "template": "a-plus-b-plus-2",
                            },
                        ]
                    },
                    "inputs": {
                        "parameters": [
                            {"name": "a", "value": "1"},
                            {"name": "b", "value": "2"},
                        ]
                    },
                    "metadata": {},
                    "name": "bettmensch-ai-inner-dag",
                    "outputs": {
                        "parameters": [
                            {
                                "name": "sum",
                                "value_from": {
                                    "parameter": "{{tasks.a-plus-b-plus-2-0.outputs.parameters.sum}}"
                                },
                            }
                        ]
                    },
                },
                {
                    "inputs": {
                        "parameters": [
                            {"default": "1", "name": "a"},
                            {"default": "2", "name": "b"},
                            {"default": "null", "name": "sum"},
                        ]
                    },
                    "metadata": {},
                    "name": "a-plus-b",
                    "outputs": {
                        "parameters": [
                            {"name": "sum", "value_from": {"path": "sum"}}
                        ]
                    },
                    "retry_strategy": {
                        "limit": "1",
                        "retry_policy": "OnError",
                    },
                    "script": {
                        "command": ["python"],
                        "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                        "image_pull_policy": "Always",
                        "name": "",
                        "resources": {
                            "limits": {"cpu": "100m", "memory": "100Mi"},
                            "requests": {"cpu": "100m", "memory": "100Mi"},
                        },
                        "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: a = json.loads(r'''{{inputs.parameters.a}}''')\nexcept: a = r'''{{inputs.parameters.a}}'''\ntry: b = json.loads(r'''{{inputs.parameters.b}}''')\nexcept: b = r'''{{inputs.parameters.b}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputParameter\nsum = OutputParameter(\"sum\")\n\ndef add_parameters(a: InputParameter=1, b: InputParameter=2, sum: OutputParameter=None) -> None:\n    \"\"\"When decorated with the bettmensch_ai.components.component decorator,\n    implements a simple addition bettmensch_ai.Component.\"\"\"\n    sum.assign(a + b)\n\nadd_parameters(a,b,sum)\n",
                    },
                },
                {
                    "inputs": {
                        "parameters": [
                            {"default": "1", "name": "a"},
                            {"default": "2", "name": "b"},
                            {"default": "null", "name": "sum"},
                        ]
                    },
                    "metadata": {},
                    "name": "a-plus-b-plus-2",
                    "outputs": {
                        "parameters": [
                            {"name": "sum", "value_from": {"path": "sum"}}
                        ]
                    },
                    "retry_strategy": {
                        "limit": "1",
                        "retry_policy": "OnError",
                    },
                    "script": {
                        "command": ["python"],
                        "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                        "image_pull_policy": "Always",
                        "name": "",
                        "resources": {
                            "limits": {"cpu": "100m", "memory": "100Mi"},
                            "requests": {"cpu": "100m", "memory": "100Mi"},
                        },
                        "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: a = json.loads(r'''{{inputs.parameters.a}}''')\nexcept: a = r'''{{inputs.parameters.a}}'''\ntry: b = json.loads(r'''{{inputs.parameters.b}}''')\nexcept: b = r'''{{inputs.parameters.b}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputParameter\nsum = OutputParameter(\"sum\")\n\ndef add_parameters(a: InputParameter=1, b: InputParameter=2, sum: OutputParameter=None) -> None:\n    \"\"\"When decorated with the bettmensch_ai.components.component decorator,\n    implements a simple addition bettmensch_ai.Component.\"\"\"\n    sum.assign(a + b)\n\nadd_parameters(a,b,sum)\n",
                    },
                },
                {
                    "dag": {
                        "tasks": [
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "a",
                                            "value": "{{workflow.parameters.a}}",
                                        },
                                        {
                                            "name": "b",
                                            "value": "{{workflow.parameters.b}}",
                                        },
                                    ]
                                },
                                "name": "bettmensch-ai-inner-dag",
                                "template": "bettmensch-ai-inner-dag",
                            }
                        ]
                    },
                    "inputs": {},
                    "metadata": {},
                    "name": "bettmensch-ai-outer-dag",
                    "outputs": {},
                },
            ],
        },
    )


@pytest.fixture
def test_hera_torch_gpu_workflow_template_model(test_datetime):

    return WorkflowTemplateModel(
        metadata={
            "creation_timestamp": test_datetime,
            "generate_name": "pipeline-test-torch-gpu-pipeline-",
            "generation": 1,
            "labels": {
                "workflows.argoproj.io/creator": "system-serviceaccount-argo-argo-server"
            },
            "managed_fields": [
                {
                    "api_version": "argoproj.io/v1alpha1",
                    "fields_type": "FieldsV1",
                    "fields_v1": {},
                    "manager": "argo",
                    "operation": "Update",
                    "time": test_datetime,
                }
            ],
            "name": "pipeline-test-torch-gpu-pipeline-7c4zp",
            "namespace": "argo",
            "resource_version": "9578",
            "uid": "612226a1-b40f-4f68-92c3-ea8a5d6b3995",
        },
        spec={
            "arguments": {
                "parameters": [{"name": "n_iter"}, {"name": "n_seconds_sleep"}]
            },
            "entrypoint": "bettmensch-ai-outer-dag",
            "templates": [
                {
                    "inputs": {},
                    "metadata": {},
                    "name": "torch-ddp-create-torch-ddp-service",
                    "outputs": {},
                    "resource": {
                        "action": "create",
                        "manifest": "apiVersion: v1\nkind: Service\nmetadata:\n  name: torch-ddp-0-{{workflow.uid}}\n  namespace: argo\n  labels:\n    workflows.argoproj.io/workflow: {{workflow.name}}\n    torch-job: torch-ddp-0\nspec:\n  clusterIP: None  # ClusterIP set to None for headless service.\n  ports:\n  - name: ddp  # Port for torchrun master<->worker node coms.\n    port: 29200\n    targetPort: 29200\n  selector:\n    workflows.argoproj.io/workflow: {{workflow.name}}\n    torch-job: torch-ddp-0\n    torch-node: '0'  # Selector for pods associated with this service.\n",
                    },
                },
                {
                    "inputs": {},
                    "metadata": {},
                    "name": "torch-ddp-delete-torch-ddp-service",
                    "outputs": {},
                    "resource": {
                        "action": "delete",
                        "flags": [
                            "service",
                            "--selector",
                            "torch-job=torch-ddp-0,workflows.argoproj.io/workflow={{workflow.name}}",
                            "-n",
                            "argo",
                        ],
                    },
                },
                {
                    "dag": {
                        "tasks": [
                            {
                                "arguments": {},
                                "name": "torch-ddp-create-torch-ddp-service",
                                "template": "torch-ddp-create-torch-ddp-service",
                            },
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "n_iter",
                                            "value": "{{inputs.parameters.n_iter}}",
                                        },
                                        {
                                            "name": "n_seconds_sleep",
                                            "value": "{{inputs.parameters.n_seconds_sleep}}",
                                        },
                                    ]
                                },
                                "depends": "torch-ddp-create-torch-ddp-service",
                                "name": "torch-ddp-0",
                                "template": "torch-ddp-0",
                            },
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "n_iter",
                                            "value": "{{inputs.parameters.n_iter}}",
                                        },
                                        {
                                            "name": "n_seconds_sleep",
                                            "value": "{{inputs.parameters.n_seconds_sleep}}",
                                        },
                                    ]
                                },
                                "depends": "torch-ddp-create-torch-ddp-service",
                                "name": "torch-ddp-0-worker-1",
                                "template": "torch-ddp-1",
                            },
                            {
                                "arguments": {},
                                "depends": "torch-ddp-0",
                                "name": "torch-ddp-delete-torch-ddp-service",
                                "template": "torch-ddp-delete-torch-ddp-service",
                            },
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "a",
                                            "value": "{{tasks.torch-ddp-0.outputs.parameters.duration}}",
                                        }
                                    ]
                                },
                                "depends": "torch-ddp-0",
                                "name": "show-duration-param-0",
                                "template": "show-duration-param",
                            },
                        ]
                    },
                    "inputs": {
                        "parameters": [
                            {"name": "n_iter"},
                            {"name": "n_seconds_sleep"},
                        ]
                    },
                    "metadata": {},
                    "name": "bettmensch-ai-inner-dag",
                    "outputs": {},
                },
                {
                    "inputs": {
                        "parameters": [
                            {"default": "100", "name": "n_iter"},
                            {"default": "10", "name": "n_seconds_sleep"},
                            {"default": "null", "name": "duration"},
                        ]
                    },
                    "metadata": {
                        "labels": {
                            "torch-job": "torch-ddp-0",
                            "torch-node": "0",
                        }
                    },
                    "name": "torch-ddp-0",
                    "outputs": {
                        "parameters": [
                            {
                                "name": "duration",
                                "value_from": {"path": "duration"},
                            }
                        ]
                    },
                    "pod_spec_patch": "topologySpreadConstraints:\n- maxSkew: 1\n  topologyKey: kubernetes.io/hostname\n  whenUnsatisfiable: DoNotSchedule\n  labelSelector:\n    matchExpressions:\n      - { key: torch-node, operator: In, values: ['0','1','2','3','4','5']}",
                    "retry_strategy": {
                        "limit": "1",
                        "retry_policy": "OnError",
                    },
                    "script": {
                        "command": ["python"],
                        "env": [
                            {"name": "NCCL_DEBUG", "value": "INFO"},
                            {
                                "name": "bettmensch_ai_torch_ddp_min_nodes",
                                "value": "2",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_max_nodes",
                                "value": "2",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_node_rank",
                                "value": "0",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_nproc_per_node",
                                "value": "1",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_max_restarts",
                                "value": "1",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_start_method",
                                "value": "fork",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_rdzv_backend",
                                "value": "static",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_rdzv_endpoint_url",
                                "value": "torch-ddp-0-{{workflow.uid}}.argo.svc.cluster.local",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_rdzv_endpoint_port",
                                "value": "29200",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_run_id",
                                "value": "1",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_tee",
                                "value": "0",
                            },
                        ],
                        "image": "bettmensch88/bettmensch.ai-pytorch:3.11-latest",
                        "image_pull_policy": "Always",
                        "name": "",
                        "ports": [
                            {
                                "container_port": 29200,
                                "name": "ddp",
                                "protocol": "TCP",
                            }
                        ],
                        "resources": {
                            "limits": {
                                "cpu": "100m",
                                "memory": "700Mi",
                                "nvidia.com/gpu": "1",
                            },
                            "requests": {
                                "cpu": "100m",
                                "memory": "700Mi",
                                "nvidia.com/gpu": "1",
                            },
                        },
                        "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: n_iter = json.loads(r'''{{inputs.parameters.n_iter}}''')\nexcept: n_iter = r'''{{inputs.parameters.n_iter}}'''\ntry: n_seconds_sleep = json.loads(r'''{{inputs.parameters.n_seconds_sleep}}''')\nexcept: n_seconds_sleep = r'''{{inputs.parameters.n_seconds_sleep}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputParameter\nduration = OutputParameter(\"duration\")\n\ndef tensor_reduce(n_iter: InputParameter=100, n_seconds_sleep: InputParameter=10, duration: OutputParameter=None) -> None:\n    \"\"\"When decorated with the torch_component decorator, implements a\n    bettmensch_ai.TorchComponent that runs a torch DDP across pods and nodes in\n    your K8s cluster.\"\"\"\n    import time\n    from datetime import datetime as dt\n    import GPUtil\n    import torch\n    import torch.distributed as dist\n    from bettmensch_ai.pipelines.component.torch_ddp import LaunchContext\n    has_gpu = torch.cuda.is_available()\n    ddp_context = LaunchContext()\n    print(f'GPU present: {has_gpu}')\n    if has_gpu:\n        dist.init_process_group(backend='nccl')\n    else:\n        dist.init_process_group(backend='gloo')\n    for i in range(1, n_iter + 1):\n        time.sleep(n_seconds_sleep)\n        GPUtil.showUtilization()\n        a = torch.tensor([ddp_context.rank])\n        print(f'{i}/{n_iter}: @{dt.now()}')\n        print(f'{i}/{n_iter}: Backend {dist.get_backend()}')\n        print(f'{i}/{n_iter}: Global world size: {ddp_context.world_size}')\n        print(f'{i}/{n_iter}: Global worker process rank: {ddp_context.rank}')\n        print(f'{i}/{n_iter}: This makes me worker process {ddp_context.rank + 1}/{ddp_context.world_size} globally!')\n        print(f'{i}/{n_iter}: Local rank of worker: {ddp_context.local_rank}')\n        print(f'{i}/{n_iter}: Local world size: {ddp_context.local_world_size}')\n        print(f'{i}/{n_iter}: This makes me worker process {ddp_context.local_rank + 1}/{ddp_context.local_world_size} locally!')\n        print(f'{i}/{n_iter}: Node/pod rank: {ddp_context.group_rank}')\n        if has_gpu:\n            device = torch.device(f'cuda:{ddp_context.local_rank}')\n            device_count = torch.cuda.device_count()\n            print(f'{i}/{n_iter}: GPU count: {device_count}')\n            device_name = torch.cuda.get_device_name(ddp_context.local_rank)\n            print(f'{i}/{n_iter}: GPU name: {device_name}')\n            device_property = torch.cuda.get_device_capability(device)\n            print(f'{i}/{n_iter}: GPU property: {device_property}')\n        else:\n            device = torch.device('cpu')\n        a_placed = a.to(device)\n        print(f'{i}/{n_iter}: Pre-`all_reduce` tensor: {a_placed}')\n        dist.all_reduce(a_placed)\n        print(f'{i}/{n_iter}: Post-`all_reduce` tensor: {a_placed}')\n        print('===================================================')\n    if duration is not None:\n        duration_seconds = n_iter * n_seconds_sleep\n        duration.assign(duration_seconds)\n\nfrom torch.distributed.elastic.multiprocessing.errors import record\n\ntensor_reduce=record(tensor_reduce)\n\nfrom bettmensch_ai.pipelines.component import as_torch_ddp\n\ntorch_ddp_decorator=as_torch_ddp()\n\ntorch_ddp_function=torch_ddp_decorator(tensor_reduce)\n\n\ntorch_ddp_function(n_iter,n_seconds_sleep,duration)",
                    },
                    "tolerations": [
                        {
                            "effect": "NoSchedule",
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                        }
                    ],
                },
                {
                    "inputs": {
                        "parameters": [
                            {"default": "100", "name": "n_iter"},
                            {"default": "10", "name": "n_seconds_sleep"},
                            {"default": "null", "name": "duration"},
                        ]
                    },
                    "metadata": {
                        "labels": {
                            "torch-job": "torch-ddp-0",
                            "torch-node": "1",
                        }
                    },
                    "name": "torch-ddp-1",
                    "outputs": {
                        "parameters": [
                            {
                                "name": "duration",
                                "value_from": {"path": "duration"},
                            }
                        ]
                    },
                    "pod_spec_patch": "topologySpreadConstraints:\n- maxSkew: 1\n  topologyKey: kubernetes.io/hostname\n  whenUnsatisfiable: DoNotSchedule\n  labelSelector:\n    matchExpressions:\n      - { key: torch-node, operator: In, values: ['0','1','2','3','4','5']}",
                    "retry_strategy": {
                        "limit": "1",
                        "retry_policy": "OnError",
                    },
                    "script": {
                        "command": ["python"],
                        "env": [
                            {"name": "NCCL_DEBUG", "value": "INFO"},
                            {
                                "name": "bettmensch_ai_torch_ddp_min_nodes",
                                "value": "2",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_max_nodes",
                                "value": "2",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_node_rank",
                                "value": "1",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_nproc_per_node",
                                "value": "1",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_max_restarts",
                                "value": "1",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_start_method",
                                "value": "fork",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_rdzv_backend",
                                "value": "static",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_rdzv_endpoint_url",
                                "value": "torch-ddp-0-{{workflow.uid}}.argo.svc.cluster.local",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_rdzv_endpoint_port",
                                "value": "29200",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_run_id",
                                "value": "1",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_tee",
                                "value": "0",
                            },
                        ],
                        "image": "bettmensch88/bettmensch.ai-pytorch:3.11-latest",
                        "image_pull_policy": "Always",
                        "name": "",
                        "resources": {
                            "limits": {
                                "cpu": "100m",
                                "memory": "700Mi",
                                "nvidia.com/gpu": "1",
                            },
                            "requests": {
                                "cpu": "100m",
                                "memory": "700Mi",
                                "nvidia.com/gpu": "1",
                            },
                        },
                        "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: n_iter = json.loads(r'''{{inputs.parameters.n_iter}}''')\nexcept: n_iter = r'''{{inputs.parameters.n_iter}}'''\ntry: n_seconds_sleep = json.loads(r'''{{inputs.parameters.n_seconds_sleep}}''')\nexcept: n_seconds_sleep = r'''{{inputs.parameters.n_seconds_sleep}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputParameter\nduration = OutputParameter(\"duration\")\n\ndef tensor_reduce(n_iter: InputParameter=100, n_seconds_sleep: InputParameter=10, duration: OutputParameter=None) -> None:\n    \"\"\"When decorated with the torch_component decorator, implements a\n    bettmensch_ai.TorchComponent that runs a torch DDP across pods and nodes in\n    your K8s cluster.\"\"\"\n    import time\n    from datetime import datetime as dt\n    import GPUtil\n    import torch\n    import torch.distributed as dist\n    from bettmensch_ai.pipelines.component.torch_ddp import LaunchContext\n    has_gpu = torch.cuda.is_available()\n    ddp_context = LaunchContext()\n    print(f'GPU present: {has_gpu}')\n    if has_gpu:\n        dist.init_process_group(backend='nccl')\n    else:\n        dist.init_process_group(backend='gloo')\n    for i in range(1, n_iter + 1):\n        time.sleep(n_seconds_sleep)\n        GPUtil.showUtilization()\n        a = torch.tensor([ddp_context.rank])\n        print(f'{i}/{n_iter}: @{dt.now()}')\n        print(f'{i}/{n_iter}: Backend {dist.get_backend()}')\n        print(f'{i}/{n_iter}: Global world size: {ddp_context.world_size}')\n        print(f'{i}/{n_iter}: Global worker process rank: {ddp_context.rank}')\n        print(f'{i}/{n_iter}: This makes me worker process {ddp_context.rank + 1}/{ddp_context.world_size} globally!')\n        print(f'{i}/{n_iter}: Local rank of worker: {ddp_context.local_rank}')\n        print(f'{i}/{n_iter}: Local world size: {ddp_context.local_world_size}')\n        print(f'{i}/{n_iter}: This makes me worker process {ddp_context.local_rank + 1}/{ddp_context.local_world_size} locally!')\n        print(f'{i}/{n_iter}: Node/pod rank: {ddp_context.group_rank}')\n        if has_gpu:\n            device = torch.device(f'cuda:{ddp_context.local_rank}')\n            device_count = torch.cuda.device_count()\n            print(f'{i}/{n_iter}: GPU count: {device_count}')\n            device_name = torch.cuda.get_device_name(ddp_context.local_rank)\n            print(f'{i}/{n_iter}: GPU name: {device_name}')\n            device_property = torch.cuda.get_device_capability(device)\n            print(f'{i}/{n_iter}: GPU property: {device_property}')\n        else:\n            device = torch.device('cpu')\n        a_placed = a.to(device)\n        print(f'{i}/{n_iter}: Pre-`all_reduce` tensor: {a_placed}')\n        dist.all_reduce(a_placed)\n        print(f'{i}/{n_iter}: Post-`all_reduce` tensor: {a_placed}')\n        print('===================================================')\n    if duration is not None:\n        duration_seconds = n_iter * n_seconds_sleep\n        duration.assign(duration_seconds)\n\nfrom torch.distributed.elastic.multiprocessing.errors import record\n\ntensor_reduce=record(tensor_reduce)\n\nfrom bettmensch_ai.pipelines.component import as_torch_ddp\n\ntorch_ddp_decorator=as_torch_ddp()\n\ntorch_ddp_function=torch_ddp_decorator(tensor_reduce)\n\n\ntorch_ddp_function(n_iter,n_seconds_sleep,duration)",
                    },
                    "tolerations": [
                        {
                            "effect": "NoSchedule",
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                        }
                    ],
                },
                {
                    "inputs": {"parameters": [{"name": "a"}]},
                    "metadata": {},
                    "name": "show-duration-param",
                    "outputs": {},
                    "retry_strategy": {
                        "limit": "1",
                        "retry_policy": "OnError",
                    },
                    "script": {
                        "command": ["python"],
                        "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                        "image_pull_policy": "Always",
                        "name": "",
                        "resources": {
                            "limits": {"cpu": "100m", "memory": "100Mi"},
                            "requests": {"cpu": "100m", "memory": "100Mi"},
                        },
                        "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: a = json.loads(r'''{{inputs.parameters.a}}''')\nexcept: a = r'''{{inputs.parameters.a}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\ndef show_parameter(a: InputParameter) -> None:\n    \"\"\"When decorated with the bettmensch_ai.components.component decorator,\n    implements a bettmensch_ai.Component that prints the values of its\n    InputParameter.\"\"\"\n    print(f'Content of input parameter a is: {a}')\n\nshow_parameter(a)\n",
                    },
                },
                {
                    "dag": {
                        "tasks": [
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "n_iter",
                                            "value": "{{workflow.parameters.n_iter}}",
                                        },
                                        {
                                            "name": "n_seconds_sleep",
                                            "value": "{{workflow.parameters.n_seconds_sleep}}",
                                        },
                                    ]
                                },
                                "name": "bettmensch-ai-inner-dag",
                                "template": "bettmensch-ai-inner-dag",
                            }
                        ]
                    },
                    "inputs": {},
                    "metadata": {},
                    "name": "bettmensch-ai-outer-dag",
                    "outputs": {},
                },
            ],
        },
    )


@pytest.fixture
def test_hera_artifact_workflow_model(test_datetime):

    return WorkflowModel(
        metadata={
            "annotations": {
                "karpenter.sh/do-not-disrupt": "true",
                "workflows.argoproj.io/pod-name-format": "v2",
            },
            "creation_timestamp": test_datetime,
            "generate_name": "pipeline-test-artifact-pipeline-jx7pb-flow-",
            "generation": 7,
            "labels": {
                "bettmensch.ai/pipeline-id": "e2e6b22b-4dfc-413d-ad43-f06a3b03cb92",
                "bettmensch.ai/pipeline-name": "pipeline-test-artifact-pipeline-jx7pb",
                "workflows.argoproj.io/completed": "true",
                "workflows.argoproj.io/creator": "system-serviceaccount-argo-argo-server",
                "workflows.argoproj.io/phase": "Succeeded",
            },
            "managed_fields": [
                {
                    "api_version": "argoproj.io/v1alpha1",
                    "fields_type": "FieldsV1",
                    "fields_v1": {},
                    "manager": "argo",
                    "operation": "Update",
                    "time": test_datetime,
                },
                {
                    "api_version": "argoproj.io/v1alpha1",
                    "fields_type": "FieldsV1",
                    "fields_v1": {},
                    "manager": "workflow-controller",
                    "operation": "Update",
                    "time": test_datetime,
                },
            ],
            "name": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d",
            "namespace": "argo",
            "resource_version": "7987",
            "uid": "e7dd825f-1f8c-4bdf-87ca-b38ae6cd773c",
        },
        spec={
            "arguments": {
                "parameters": [
                    {"name": "a", "value": "First integration test value a"}
                ]
            },
            "workflow_template_ref": {
                "name": "pipeline-test-artifact-pipeline-jx7pb"
            },
        },
        status={
            "artifact_gc_status": {"not_specified": True},
            "artifact_repository_ref": {
                "artifact_repository": {
                    "s3": {
                        "bucket": "bettmensch-ai-artifact-repository",
                        "endpoint": "s3.us-east-2.amazonaws.com",
                        "insecure": True,
                        "key_format": "argo-workflows/{{workflow.name}}/{{pod.name}}",
                    }
                },
                "config_map": "artifact-repositories",
                "key": "bettmensch-ai-artifact-repository",
                "namespace": "argo",
            },
            "conditions": [
                {"status": "False", "type": "PodRunning"},
                {"status": "True", "type": "Completed"},
            ],
            "finished_at": test_datetime,
            "nodes": {
                "pipeline-test-artifact-pipeline-jx7pb-flow-md47d": {
                    "children": [
                        "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-4230836876"
                    ],
                    "display_name": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d",
                    "finished_at": test_datetime,
                    "id": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d",
                    "name": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d",
                    "outbound_nodes": [
                        "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-1613118188"
                    ],
                    "phase": "Succeeded",
                    "progress": "2/2",
                    "resources_duration": {"cpu": 2, "memory": 68},
                    "started_at": test_datetime,
                    "template_name": "bettmensch-ai-outer-dag",
                    "template_scope": "local/",
                    "type": "DAG",
                },
                "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-1074722518": {
                    "boundary_id": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-4230836876",
                    "children": [
                        "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-170779741"
                    ],
                    "display_name": "convert-to-artifact-0(0)",
                    "finished_at": test_datetime,
                    "host_node_name": "ip-10-0-48-85.us-east-2.compute.internal",
                    "id": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-1074722518",
                    "inputs": {
                        "parameters": [
                            {
                                "name": "a",
                                "value": "First integration test value a",
                            },
                            {
                                "default": "null",
                                "name": "a_art",
                                "value": "null",
                            },
                        ]
                    },
                    "name": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d.bettmensch-ai-inner-dag.convert-to-artifact-0(0)",
                    "node_flag": {"retried": True},
                    "outputs": {
                        "artifacts": [
                            {
                                "name": "a_art",
                                "path": "a_art",
                                "s3": {
                                    "key": "argo-workflows/pipeline-test-artifact-pipeline-jx7pb-flow-md47d/pipeline-test-artifact-pipeline-jx7pb-flow-md47d-convert-to-artifact-1074722518/a_art.tgz"
                                },
                            }
                        ],
                        "exit_code": "0",
                    },
                    "phase": "Succeeded",
                    "progress": "1/1",
                    "resources_duration": {"cpu": 1, "memory": 43},
                    "started_at": test_datetime,
                    "template_name": "convert-to-artifact",
                    "template_scope": "local/",
                    "type": "Pod",
                },
                "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-1613118188": {
                    "boundary_id": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-4230836876",
                    "display_name": "show-artifact-0(0)",
                    "finished_at": test_datetime,
                    "host_node_name": "ip-10-0-49-235.us-east-2.compute.internal",
                    "id": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-1613118188",
                    "inputs": {
                        "artifacts": [
                            {
                                "name": "a",
                                "path": "a",
                                "s3": {
                                    "key": "argo-workflows/pipeline-test-artifact-pipeline-jx7pb-flow-md47d/pipeline-test-artifact-pipeline-jx7pb-flow-md47d-convert-to-artifact-1074722518/a_art.tgz"
                                },
                            }
                        ],
                        "parameters": [
                            {"default": "null", "name": "b", "value": "null"}
                        ],
                    },
                    "name": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d.bettmensch-ai-inner-dag.show-artifact-0(0)",
                    "node_flag": {"retried": True},
                    "outputs": {
                        "artifacts": [
                            {
                                "name": "b",
                                "path": "b",
                                "s3": {
                                    "key": "argo-workflows/pipeline-test-artifact-pipeline-jx7pb-flow-md47d/pipeline-test-artifact-pipeline-jx7pb-flow-md47d-show-artifact-1613118188/b.tgz"
                                },
                            }
                        ],
                        "exit_code": "0",
                    },
                    "phase": "Succeeded",
                    "progress": "1/1",
                    "resources_duration": {"cpu": 1, "memory": 25},
                    "started_at": test_datetime,
                    "template_name": "show-artifact",
                    "template_scope": "local/",
                    "type": "Pod",
                },
                "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-170779741": {
                    "boundary_id": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-4230836876",
                    "children": [
                        "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-1613118188"
                    ],
                    "display_name": "show-artifact-0",
                    "finished_at": test_datetime,
                    "id": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-170779741",
                    "inputs": {
                        "artifacts": [
                            {
                                "name": "a",
                                "path": "a",
                                "s3": {
                                    "key": "argo-workflows/pipeline-test-artifact-pipeline-jx7pb-flow-md47d/pipeline-test-artifact-pipeline-jx7pb-flow-md47d-convert-to-artifact-1074722518/a_art.tgz"
                                },
                            }
                        ],
                        "parameters": [
                            {"default": "null", "name": "b", "value": "null"}
                        ],
                    },
                    "name": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d.bettmensch-ai-inner-dag.show-artifact-0",
                    "outputs": {
                        "artifacts": [
                            {
                                "name": "b",
                                "path": "b",
                                "s3": {
                                    "key": "argo-workflows/pipeline-test-artifact-pipeline-jx7pb-flow-md47d/pipeline-test-artifact-pipeline-jx7pb-flow-md47d-show-artifact-1613118188/b.tgz"
                                },
                            }
                        ],
                        "exit_code": "0",
                    },
                    "phase": "Succeeded",
                    "progress": "1/1",
                    "resources_duration": {"cpu": 1, "memory": 25},
                    "started_at": test_datetime,
                    "template_name": "show-artifact",
                    "template_scope": "local/",
                    "type": "Retry",
                },
                "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-1834257243": {
                    "boundary_id": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-4230836876",
                    "children": [
                        "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-1074722518"
                    ],
                    "display_name": "convert-to-artifact-0",
                    "finished_at": test_datetime,
                    "id": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-1834257243",
                    "inputs": {
                        "parameters": [
                            {
                                "name": "a",
                                "value": "First integration test value a",
                            },
                            {
                                "default": "null",
                                "name": "a_art",
                                "value": "null",
                            },
                        ]
                    },
                    "name": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d.bettmensch-ai-inner-dag.convert-to-artifact-0",
                    "outputs": {
                        "artifacts": [
                            {
                                "name": "a_art",
                                "path": "a_art",
                                "s3": {
                                    "key": "argo-workflows/pipeline-test-artifact-pipeline-jx7pb-flow-md47d/pipeline-test-artifact-pipeline-jx7pb-flow-md47d-convert-to-artifact-1074722518/a_art.tgz"
                                },
                            }
                        ],
                        "exit_code": "0",
                    },
                    "phase": "Succeeded",
                    "progress": "2/2",
                    "resources_duration": {"cpu": 2, "memory": 68},
                    "started_at": test_datetime,
                    "template_name": "convert-to-artifact",
                    "template_scope": "local/",
                    "type": "Retry",
                },
                "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-4230836876": {
                    "boundary_id": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d",
                    "children": [
                        "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-1834257243"
                    ],
                    "display_name": "bettmensch-ai-inner-dag",
                    "finished_at": test_datetime,
                    "id": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-4230836876",
                    "inputs": {
                        "parameters": [
                            {
                                "name": "a",
                                "value": "First integration test value a",
                            }
                        ]
                    },
                    "name": "pipeline-test-artifact-pipeline-jx7pb-flow-md47d.bettmensch-ai-inner-dag",
                    "outbound_nodes": [
                        "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-1613118188"
                    ],
                    "outputs": {
                        "artifacts": [
                            {
                                "name": "b",
                                "path": "b",
                                "s3": {
                                    "key": "argo-workflows/pipeline-test-artifact-pipeline-jx7pb-flow-md47d/pipeline-test-artifact-pipeline-jx7pb-flow-md47d-show-artifact-1613118188/b.tgz"
                                },
                            }
                        ]
                    },
                    "phase": "Succeeded",
                    "progress": "2/2",
                    "resources_duration": {"cpu": 2, "memory": 68},
                    "started_at": test_datetime,
                    "template_name": "bettmensch-ai-inner-dag",
                    "template_scope": "local/",
                    "type": "DAG",
                },
            },
            "phase": "Succeeded",
            "progress": "2/2",
            "resources_duration": {"cpu": 2, "memory": 68},
            "started_at": test_datetime,
            "stored_templates": {
                "namespaced/pipeline-test-artifact-pipeline-jx7pb/bettmensch-ai-inner-dag": {
                    "dag": {
                        "tasks": [
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "a",
                                            "value": "{{inputs.parameters.a}}",
                                        }
                                    ]
                                },
                                "name": "convert-to-artifact-0",
                                "template": "convert-to-artifact",
                            },
                            {
                                "arguments": {
                                    "artifacts": [
                                        {
                                            "from_": "{{tasks.convert-to-artifact-0.outputs.artifacts.a_art}}",
                                            "name": "a",
                                        }
                                    ]
                                },
                                "depends": "convert-to-artifact-0",
                                "name": "show-artifact-0",
                                "template": "show-artifact",
                            },
                        ]
                    },
                    "inputs": {
                        "parameters": [{"name": "a", "value": "Param A"}]
                    },
                    "metadata": {},
                    "name": "bettmensch-ai-inner-dag",
                    "outputs": {
                        "artifacts": [
                            {
                                "from_": "{{tasks.show-artifact-0.outputs.artifacts.b}}",
                                "name": "b",
                            }
                        ]
                    },
                },
                "namespaced/pipeline-test-artifact-pipeline-jx7pb/bettmensch-ai-outer-dag": {
                    "dag": {
                        "tasks": [
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "a",
                                            "value": "{{workflow.parameters.a}}",
                                        }
                                    ]
                                },
                                "name": "bettmensch-ai-inner-dag",
                                "template": "bettmensch-ai-inner-dag",
                            }
                        ]
                    },
                    "inputs": {},
                    "metadata": {},
                    "name": "bettmensch-ai-outer-dag",
                    "outputs": {},
                },
                "namespaced/pipeline-test-artifact-pipeline-jx7pb/convert-to-artifact": {
                    "inputs": {
                        "parameters": [
                            {"name": "a"},
                            {"default": "null", "name": "a_art"},
                        ]
                    },
                    "metadata": {},
                    "name": "convert-to-artifact",
                    "outputs": {
                        "artifacts": [{"name": "a_art", "path": "a_art"}]
                    },
                    "retry_strategy": {
                        "limit": "1",
                        "retry_policy": "OnError",
                    },
                    "script": {
                        "command": ["python"],
                        "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                        "image_pull_policy": "Always",
                        "name": "",
                        "resources": {
                            "limits": {"cpu": "100m", "memory": "100Mi"},
                            "requests": {"cpu": "100m", "memory": "100Mi"},
                        },
                        "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: a = json.loads(r'''{{inputs.parameters.a}}''')\nexcept: a = r'''{{inputs.parameters.a}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputArtifact\na_art = OutputArtifact(\"a_art\")\n\ndef convert_to_artifact(a: InputParameter, a_art: OutputArtifact=None) -> None:\n    \"\"\"When decorated with the bettmensch_ai.components.component decorator,\n    implements a bettmensch_ai.Component that converts its InputParameter into\n    an OutputArtifact.\"\"\"\n    with open(a_art.path, 'w') as a_art_file:\n        a_art_file.write(str(a))\n\nconvert_to_artifact(a,a_art)\n",
                    },
                },
                "namespaced/pipeline-test-artifact-pipeline-jx7pb/show-artifact": {
                    "inputs": {
                        "artifacts": [{"name": "a", "path": "a"}],
                        "parameters": [{"default": "null", "name": "b"}],
                    },
                    "metadata": {},
                    "name": "show-artifact",
                    "outputs": {"artifacts": [{"name": "b", "path": "b"}]},
                    "retry_strategy": {
                        "limit": "1",
                        "retry_policy": "OnError",
                    },
                    "script": {
                        "command": ["python"],
                        "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                        "image_pull_policy": "Always",
                        "name": "",
                        "resources": {
                            "limits": {"cpu": "100m", "memory": "100Mi"},
                            "requests": {"cpu": "100m", "memory": "100Mi"},
                        },
                        "source": 'import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import InputArtifact\na = InputArtifact("a")\n\nfrom bettmensch_ai.pipelines.io import OutputArtifact\nb = OutputArtifact("b")\n\ndef show_artifact(a: InputArtifact, b: OutputArtifact=None) -> None:\n    """When decorated with the bettmensch_ai.components.component decorator,\n    implements a bettmensch_ai.Component that prints the values of its\n    InputArtifact."""\n    with open(a.path, \'r\') as a_art_file:\n        a_content = a_art_file.read()\n    print(f\'Content of input artifact a: {a_content}\')\n    with open(b.path, \'w\') as b_art_file:\n        b_art_file.write(str(a_content))\n\nshow_artifact(a,b)\n',
                    },
                },
            },
            "stored_workflow_template_spec": {
                "arguments": {
                    "parameters": [
                        {
                            "name": "a",
                            "value": "First integration test value a",
                        }
                    ]
                },
                "entrypoint": "bettmensch-ai-outer-dag",
                "service_account_name": "argo-workflow",
                "templates": [
                    {
                        "dag": {
                            "tasks": [
                                {
                                    "arguments": {
                                        "parameters": [
                                            {
                                                "name": "a",
                                                "value": "{{inputs.parameters.a}}",
                                            }
                                        ]
                                    },
                                    "name": "convert-to-artifact-0",
                                    "template": "convert-to-artifact",
                                },
                                {
                                    "arguments": {
                                        "artifacts": [
                                            {
                                                "from_": "{{tasks.convert-to-artifact-0.outputs.artifacts.a_art}}",
                                                "name": "a",
                                            }
                                        ]
                                    },
                                    "depends": "convert-to-artifact-0",
                                    "name": "show-artifact-0",
                                    "template": "show-artifact",
                                },
                            ]
                        },
                        "inputs": {
                            "parameters": [{"name": "a", "value": "Param A"}]
                        },
                        "metadata": {},
                        "name": "bettmensch-ai-inner-dag",
                        "outputs": {
                            "artifacts": [
                                {
                                    "from_": "{{tasks.show-artifact-0.outputs.artifacts.b}}",
                                    "name": "b",
                                }
                            ]
                        },
                    },
                    {
                        "inputs": {
                            "parameters": [
                                {"name": "a"},
                                {"default": "null", "name": "a_art"},
                            ]
                        },
                        "metadata": {},
                        "name": "convert-to-artifact",
                        "outputs": {
                            "artifacts": [{"name": "a_art", "path": "a_art"}]
                        },
                        "retry_strategy": {
                            "limit": "1",
                            "retry_policy": "OnError",
                        },
                        "script": {
                            "command": ["python"],
                            "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                            "image_pull_policy": "Always",
                            "name": "",
                            "resources": {
                                "limits": {"cpu": "100m", "memory": "100Mi"},
                                "requests": {"cpu": "100m", "memory": "100Mi"},
                            },
                            "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: a = json.loads(r'''{{inputs.parameters.a}}''')\nexcept: a = r'''{{inputs.parameters.a}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputArtifact\na_art = OutputArtifact(\"a_art\")\n\ndef convert_to_artifact(a: InputParameter, a_art: OutputArtifact=None) -> None:\n    \"\"\"When decorated with the bettmensch_ai.components.component decorator,\n    implements a bettmensch_ai.Component that converts its InputParameter into\n    an OutputArtifact.\"\"\"\n    with open(a_art.path, 'w') as a_art_file:\n        a_art_file.write(str(a))\n\nconvert_to_artifact(a,a_art)\n",
                        },
                    },
                    {
                        "inputs": {
                            "artifacts": [{"name": "a", "path": "a"}],
                            "parameters": [{"default": "null", "name": "b"}],
                        },
                        "metadata": {},
                        "name": "show-artifact",
                        "outputs": {"artifacts": [{"name": "b", "path": "b"}]},
                        "retry_strategy": {
                            "limit": "1",
                            "retry_policy": "OnError",
                        },
                        "script": {
                            "command": ["python"],
                            "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                            "image_pull_policy": "Always",
                            "name": "",
                            "resources": {
                                "limits": {"cpu": "100m", "memory": "100Mi"},
                                "requests": {"cpu": "100m", "memory": "100Mi"},
                            },
                            "source": 'import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import InputArtifact\na = InputArtifact("a")\n\nfrom bettmensch_ai.pipelines.io import OutputArtifact\nb = OutputArtifact("b")\n\ndef show_artifact(a: InputArtifact, b: OutputArtifact=None) -> None:\n    """When decorated with the bettmensch_ai.components.component decorator,\n    implements a bettmensch_ai.Component that prints the values of its\n    InputArtifact."""\n    with open(a.path, \'r\') as a_art_file:\n        a_content = a_art_file.read()\n    print(f\'Content of input artifact a: {a_content}\')\n    with open(b.path, \'w\') as b_art_file:\n        b_art_file.write(str(a_content))\n\nshow_artifact(a,b)\n',
                        },
                    },
                    {
                        "dag": {
                            "tasks": [
                                {
                                    "arguments": {
                                        "parameters": [
                                            {
                                                "name": "a",
                                                "value": "{{workflow.parameters.a}}",
                                            }
                                        ]
                                    },
                                    "name": "bettmensch-ai-inner-dag",
                                    "template": "bettmensch-ai-inner-dag",
                                }
                            ]
                        },
                        "inputs": {},
                        "metadata": {},
                        "name": "bettmensch-ai-outer-dag",
                        "outputs": {},
                    },
                ],
                "workflow_template_ref": {
                    "name": "pipeline-test-artifact-pipeline-jx7pb"
                },
            },
            "task_results_completion_status": {
                "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-1074722518": True,
                "pipeline-test-artifact-pipeline-jx7pb-flow-md47d-1613118188": True,
            },
        },
    )


@pytest.fixture
def test_hera_parameter_workflow_model(test_datetime):

    return WorkflowModel(
        metadata={
            "annotations": {
                "karpenter.sh/do-not-disrupt": "true",
                "workflows.argoproj.io/pod-name-format": "v2",
            },
            "creation_timestamp": test_datetime,
            "generate_name": "pipeline-test-parameter-pipeline-c877j-flow-",
            "generation": 7,
            "labels": {
                "bettmensch.ai/pipeline-id": "d2715290-865d-4776-84c4-776632cd7159",
                "bettmensch.ai/pipeline-name": "pipeline-test-parameter-pipeline-c877j",
                "workflows.argoproj.io/completed": "true",
                "workflows.argoproj.io/creator": "system-serviceaccount-argo-argo-server",
                "workflows.argoproj.io/phase": "Succeeded",
            },
            "managed_fields": [
                {
                    "api_version": "argoproj.io/v1alpha1",
                    "fields_type": "FieldsV1",
                    "fields_v1": {},
                    "manager": "argo",
                    "operation": "Update",
                    "time": test_datetime,
                },
                {
                    "api_version": "argoproj.io/v1alpha1",
                    "fields_type": "FieldsV1",
                    "fields_v1": {},
                    "manager": "workflow-controller",
                    "operation": "Update",
                    "time": test_datetime,
                },
            ],
            "name": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn",
            "namespace": "argo",
            "resource_version": "8018",
            "uid": "f4623367-e5c2-4ba7-9a7a-633c55314421",
        },
        spec={
            "arguments": {
                "parameters": [
                    {"name": "a", "value": "-100"},
                    {"name": "b", "value": "100"},
                ]
            },
            "workflow_template_ref": {
                "name": "pipeline-test-parameter-pipeline-c877j"
            },
        },
        status={
            "artifact_gc_status": {"not_specified": True},
            "artifact_repository_ref": {
                "artifact_repository": {
                    "s3": {
                        "bucket": "bettmensch-ai-artifact-repository",
                        "endpoint": "s3.us-east-2.amazonaws.com",
                        "insecure": True,
                        "key_format": "argo-workflows/{{workflow.name}}/{{pod.name}}",
                    }
                },
                "config_map": "artifact-repositories",
                "key": "bettmensch-ai-artifact-repository",
                "namespace": "argo",
            },
            "conditions": [
                {"status": "False", "type": "PodRunning"},
                {"status": "True", "type": "Completed"},
            ],
            "finished_at": test_datetime,
            "nodes": {
                "pipeline-test-parameter-pipeline-c877j-flow-tfgmn": {
                    "children": [
                        "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-1140354891"
                    ],
                    "display_name": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn",
                    "finished_at": test_datetime,
                    "id": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn",
                    "name": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn",
                    "outbound_nodes": [
                        "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-4267990770"
                    ],
                    "phase": "Succeeded",
                    "progress": "2/2",
                    "resources_duration": {"cpu": 2, "memory": 54},
                    "started_at": test_datetime,
                    "template_name": "bettmensch-ai-outer-dag",
                    "template_scope": "local/",
                    "type": "DAG",
                },
                "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-1140354891": {
                    "boundary_id": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn",
                    "children": [
                        "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-3695553323"
                    ],
                    "display_name": "bettmensch-ai-inner-dag",
                    "finished_at": test_datetime,
                    "id": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-1140354891",
                    "inputs": {
                        "parameters": [
                            {"name": "a", "value": "-100"},
                            {"name": "b", "value": "100"},
                        ]
                    },
                    "name": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn.bettmensch-ai-inner-dag",
                    "outbound_nodes": [
                        "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-4267990770"
                    ],
                    "outputs": {"parameters": [{"name": "sum", "value": "2"}]},
                    "phase": "Succeeded",
                    "progress": "2/2",
                    "resources_duration": {"cpu": 2, "memory": 54},
                    "started_at": test_datetime,
                    "template_name": "bettmensch-ai-inner-dag",
                    "template_scope": "local/",
                    "type": "DAG",
                },
                "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-1412890278": {
                    "boundary_id": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-1140354891",
                    "children": [
                        "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-1697420911"
                    ],
                    "display_name": "a-plus-b-0(0)",
                    "finished_at": test_datetime,
                    "host_node_name": "ip-10-0-49-235.us-east-2.compute.internal",
                    "id": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-1412890278",
                    "inputs": {
                        "parameters": [
                            {"default": "1", "name": "a", "value": "-100"},
                            {"default": "2", "name": "b", "value": "100"},
                            {
                                "default": "null",
                                "name": "sum",
                                "value": "null",
                            },
                        ]
                    },
                    "name": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn.bettmensch-ai-inner-dag.a-plus-b-0(0)",
                    "node_flag": {"retried": True},
                    "outputs": {
                        "exit_code": "0",
                        "parameters": [
                            {
                                "name": "sum",
                                "value": "0",
                                "value_from": {"path": "sum"},
                            }
                        ],
                    },
                    "phase": "Succeeded",
                    "progress": "1/1",
                    "resources_duration": {"cpu": 1, "memory": 28},
                    "started_at": test_datetime,
                    "template_name": "a-plus-b",
                    "template_scope": "local/",
                    "type": "Pod",
                },
                "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-1697420911": {
                    "boundary_id": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-1140354891",
                    "children": [
                        "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-4267990770"
                    ],
                    "display_name": "a-plus-b-plus-2-0",
                    "finished_at": test_datetime,
                    "id": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-1697420911",
                    "inputs": {
                        "parameters": [
                            {"default": "1", "name": "a", "value": "0"},
                            {"default": "2", "name": "b", "value": "2"},
                            {
                                "default": "null",
                                "name": "sum",
                                "value": "null",
                            },
                        ]
                    },
                    "name": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn.bettmensch-ai-inner-dag.a-plus-b-plus-2-0",
                    "outputs": {
                        "exit_code": "0",
                        "parameters": [
                            {
                                "name": "sum",
                                "value": "2",
                                "value_from": {"path": "sum"},
                            }
                        ],
                    },
                    "phase": "Succeeded",
                    "progress": "1/1",
                    "resources_duration": {"cpu": 1, "memory": 26},
                    "started_at": test_datetime,
                    "template_name": "a-plus-b-plus-2",
                    "template_scope": "local/",
                    "type": "Retry",
                },
                "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-3695553323": {
                    "boundary_id": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-1140354891",
                    "children": [
                        "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-1412890278"
                    ],
                    "display_name": "a-plus-b-0",
                    "finished_at": test_datetime,
                    "id": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-3695553323",
                    "inputs": {
                        "parameters": [
                            {"default": "1", "name": "a", "value": "-100"},
                            {"default": "2", "name": "b", "value": "100"},
                            {
                                "default": "null",
                                "name": "sum",
                                "value": "null",
                            },
                        ]
                    },
                    "name": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn.bettmensch-ai-inner-dag.a-plus-b-0",
                    "outputs": {
                        "exit_code": "0",
                        "parameters": [
                            {
                                "name": "sum",
                                "value": "0",
                                "value_from": {"path": "sum"},
                            }
                        ],
                    },
                    "phase": "Succeeded",
                    "progress": "2/2",
                    "resources_duration": {"cpu": 2, "memory": 54},
                    "started_at": test_datetime,
                    "template_name": "a-plus-b",
                    "template_scope": "local/",
                    "type": "Retry",
                },
                "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-4267990770": {
                    "boundary_id": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-1140354891",
                    "display_name": "a-plus-b-plus-2-0(0)",
                    "finished_at": test_datetime,
                    "host_node_name": "ip-10-0-48-85.us-east-2.compute.internal",
                    "id": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-4267990770",
                    "inputs": {
                        "parameters": [
                            {"default": "1", "name": "a", "value": "0"},
                            {"default": "2", "name": "b", "value": "2"},
                            {
                                "default": "null",
                                "name": "sum",
                                "value": "null",
                            },
                        ]
                    },
                    "name": "pipeline-test-parameter-pipeline-c877j-flow-tfgmn.bettmensch-ai-inner-dag.a-plus-b-plus-2-0(0)",
                    "node_flag": {"retried": True},
                    "outputs": {
                        "exit_code": "0",
                        "parameters": [
                            {
                                "name": "sum",
                                "value": "2",
                                "value_from": {"path": "sum"},
                            }
                        ],
                    },
                    "phase": "Succeeded",
                    "progress": "1/1",
                    "resources_duration": {"cpu": 1, "memory": 26},
                    "started_at": test_datetime,
                    "template_name": "a-plus-b-plus-2",
                    "template_scope": "local/",
                    "type": "Pod",
                },
            },
            "phase": "Succeeded",
            "progress": "2/2",
            "resources_duration": {"cpu": 2, "memory": 54},
            "started_at": test_datetime,
            "stored_templates": {
                "namespaced/pipeline-test-parameter-pipeline-c877j/a-plus-b": {
                    "inputs": {
                        "parameters": [
                            {"default": "1", "name": "a"},
                            {"default": "2", "name": "b"},
                            {"default": "null", "name": "sum"},
                        ]
                    },
                    "metadata": {},
                    "name": "a-plus-b",
                    "outputs": {
                        "parameters": [
                            {"name": "sum", "value_from": {"path": "sum"}}
                        ]
                    },
                    "retry_strategy": {
                        "limit": "1",
                        "retry_policy": "OnError",
                    },
                    "script": {
                        "command": ["python"],
                        "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                        "image_pull_policy": "Always",
                        "name": "",
                        "resources": {
                            "limits": {"cpu": "100m", "memory": "100Mi"},
                            "requests": {"cpu": "100m", "memory": "100Mi"},
                        },
                        "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: a = json.loads(r'''{{inputs.parameters.a}}''')\nexcept: a = r'''{{inputs.parameters.a}}'''\ntry: b = json.loads(r'''{{inputs.parameters.b}}''')\nexcept: b = r'''{{inputs.parameters.b}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputParameter\nsum = OutputParameter(\"sum\")\n\ndef add_parameters(a: InputParameter=1, b: InputParameter=2, sum: OutputParameter=None) -> None:\n    \"\"\"When decorated with the bettmensch_ai.components.component decorator,\n    implements a simple addition bettmensch_ai.Component.\"\"\"\n    sum.assign(a + b)\n\nadd_parameters(a,b,sum)\n",
                    },
                },
                "namespaced/pipeline-test-parameter-pipeline-c877j/a-plus-b-plus-2": {
                    "inputs": {
                        "parameters": [
                            {"default": "1", "name": "a"},
                            {"default": "2", "name": "b"},
                            {"default": "null", "name": "sum"},
                        ]
                    },
                    "metadata": {},
                    "name": "a-plus-b-plus-2",
                    "outputs": {
                        "parameters": [
                            {"name": "sum", "value_from": {"path": "sum"}}
                        ]
                    },
                    "retry_strategy": {
                        "limit": "1",
                        "retry_policy": "OnError",
                    },
                    "script": {
                        "command": ["python"],
                        "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                        "image_pull_policy": "Always",
                        "name": "",
                        "resources": {
                            "limits": {"cpu": "100m", "memory": "100Mi"},
                            "requests": {"cpu": "100m", "memory": "100Mi"},
                        },
                        "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: a = json.loads(r'''{{inputs.parameters.a}}''')\nexcept: a = r'''{{inputs.parameters.a}}'''\ntry: b = json.loads(r'''{{inputs.parameters.b}}''')\nexcept: b = r'''{{inputs.parameters.b}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputParameter\nsum = OutputParameter(\"sum\")\n\ndef add_parameters(a: InputParameter=1, b: InputParameter=2, sum: OutputParameter=None) -> None:\n    \"\"\"When decorated with the bettmensch_ai.components.component decorator,\n    implements a simple addition bettmensch_ai.Component.\"\"\"\n    sum.assign(a + b)\n\nadd_parameters(a,b,sum)\n",
                    },
                },
                "namespaced/pipeline-test-parameter-pipeline-c877j/bettmensch-ai-inner-dag": {
                    "dag": {
                        "tasks": [
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "a",
                                            "value": "{{inputs.parameters.a}}",
                                        },
                                        {
                                            "name": "b",
                                            "value": "{{inputs.parameters.b}}",
                                        },
                                    ]
                                },
                                "name": "a-plus-b-0",
                                "template": "a-plus-b",
                            },
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "a",
                                            "value": "{{tasks.a-plus-b-0.outputs.parameters.sum}}",
                                        },
                                        {"name": "b", "value": "2"},
                                    ]
                                },
                                "depends": "a-plus-b-0",
                                "name": "a-plus-b-plus-2-0",
                                "template": "a-plus-b-plus-2",
                            },
                        ]
                    },
                    "inputs": {
                        "parameters": [
                            {"name": "a", "value": "1"},
                            {"name": "b", "value": "2"},
                        ]
                    },
                    "metadata": {},
                    "name": "bettmensch-ai-inner-dag",
                    "outputs": {
                        "parameters": [
                            {
                                "name": "sum",
                                "value_from": {
                                    "parameter": "{{tasks.a-plus-b-plus-2-0.outputs.parameters.sum}}"
                                },
                            }
                        ]
                    },
                },
                "namespaced/pipeline-test-parameter-pipeline-c877j/bettmensch-ai-outer-dag": {
                    "dag": {
                        "tasks": [
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "a",
                                            "value": "{{workflow.parameters.a}}",
                                        },
                                        {
                                            "name": "b",
                                            "value": "{{workflow.parameters.b}}",
                                        },
                                    ]
                                },
                                "name": "bettmensch-ai-inner-dag",
                                "template": "bettmensch-ai-inner-dag",
                            }
                        ]
                    },
                    "inputs": {},
                    "metadata": {},
                    "name": "bettmensch-ai-outer-dag",
                    "outputs": {},
                },
            },
            "stored_workflow_template_spec": {
                "arguments": {
                    "parameters": [
                        {"name": "a", "value": "-100"},
                        {"name": "b", "value": "100"},
                    ]
                },
                "entrypoint": "bettmensch-ai-outer-dag",
                "service_account_name": "argo-workflow",
                "templates": [
                    {
                        "dag": {
                            "tasks": [
                                {
                                    "arguments": {
                                        "parameters": [
                                            {
                                                "name": "a",
                                                "value": "{{inputs.parameters.a}}",
                                            },
                                            {
                                                "name": "b",
                                                "value": "{{inputs.parameters.b}}",
                                            },
                                        ]
                                    },
                                    "name": "a-plus-b-0",
                                    "template": "a-plus-b",
                                },
                                {
                                    "arguments": {
                                        "parameters": [
                                            {
                                                "name": "a",
                                                "value": "{{tasks.a-plus-b-0.outputs.parameters.sum}}",
                                            },
                                            {"name": "b", "value": "2"},
                                        ]
                                    },
                                    "depends": "a-plus-b-0",
                                    "name": "a-plus-b-plus-2-0",
                                    "template": "a-plus-b-plus-2",
                                },
                            ]
                        },
                        "inputs": {
                            "parameters": [
                                {"name": "a", "value": "1"},
                                {"name": "b", "value": "2"},
                            ]
                        },
                        "metadata": {},
                        "name": "bettmensch-ai-inner-dag",
                        "outputs": {
                            "parameters": [
                                {
                                    "name": "sum",
                                    "value_from": {
                                        "parameter": "{{tasks.a-plus-b-plus-2-0.outputs.parameters.sum}}"
                                    },
                                }
                            ]
                        },
                    },
                    {
                        "inputs": {
                            "parameters": [
                                {"default": "1", "name": "a"},
                                {"default": "2", "name": "b"},
                                {"default": "null", "name": "sum"},
                            ]
                        },
                        "metadata": {},
                        "name": "a-plus-b",
                        "outputs": {
                            "parameters": [
                                {"name": "sum", "value_from": {"path": "sum"}}
                            ]
                        },
                        "retry_strategy": {
                            "limit": "1",
                            "retry_policy": "OnError",
                        },
                        "script": {
                            "command": ["python"],
                            "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                            "image_pull_policy": "Always",
                            "name": "",
                            "resources": {
                                "limits": {"cpu": "100m", "memory": "100Mi"},
                                "requests": {"cpu": "100m", "memory": "100Mi"},
                            },
                            "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: a = json.loads(r'''{{inputs.parameters.a}}''')\nexcept: a = r'''{{inputs.parameters.a}}'''\ntry: b = json.loads(r'''{{inputs.parameters.b}}''')\nexcept: b = r'''{{inputs.parameters.b}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputParameter\nsum = OutputParameter(\"sum\")\n\ndef add_parameters(a: InputParameter=1, b: InputParameter=2, sum: OutputParameter=None) -> None:\n    \"\"\"When decorated with the bettmensch_ai.components.component decorator,\n    implements a simple addition bettmensch_ai.Component.\"\"\"\n    sum.assign(a + b)\n\nadd_parameters(a,b,sum)\n",
                        },
                    },
                    {
                        "inputs": {
                            "parameters": [
                                {"default": "1", "name": "a"},
                                {"default": "2", "name": "b"},
                                {"default": "null", "name": "sum"},
                            ]
                        },
                        "metadata": {},
                        "name": "a-plus-b-plus-2",
                        "outputs": {
                            "parameters": [
                                {"name": "sum", "value_from": {"path": "sum"}}
                            ]
                        },
                        "retry_strategy": {
                            "limit": "1",
                            "retry_policy": "OnError",
                        },
                        "script": {
                            "command": ["python"],
                            "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                            "image_pull_policy": "Always",
                            "name": "",
                            "resources": {
                                "limits": {"cpu": "100m", "memory": "100Mi"},
                                "requests": {"cpu": "100m", "memory": "100Mi"},
                            },
                            "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: a = json.loads(r'''{{inputs.parameters.a}}''')\nexcept: a = r'''{{inputs.parameters.a}}'''\ntry: b = json.loads(r'''{{inputs.parameters.b}}''')\nexcept: b = r'''{{inputs.parameters.b}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputParameter\nsum = OutputParameter(\"sum\")\n\ndef add_parameters(a: InputParameter=1, b: InputParameter=2, sum: OutputParameter=None) -> None:\n    \"\"\"When decorated with the bettmensch_ai.components.component decorator,\n    implements a simple addition bettmensch_ai.Component.\"\"\"\n    sum.assign(a + b)\n\nadd_parameters(a,b,sum)\n",
                        },
                    },
                    {
                        "dag": {
                            "tasks": [
                                {
                                    "arguments": {
                                        "parameters": [
                                            {
                                                "name": "a",
                                                "value": "{{workflow.parameters.a}}",
                                            },
                                            {
                                                "name": "b",
                                                "value": "{{workflow.parameters.b}}",
                                            },
                                        ]
                                    },
                                    "name": "bettmensch-ai-inner-dag",
                                    "template": "bettmensch-ai-inner-dag",
                                }
                            ]
                        },
                        "inputs": {},
                        "metadata": {},
                        "name": "bettmensch-ai-outer-dag",
                        "outputs": {},
                    },
                ],
                "workflow_template_ref": {
                    "name": "pipeline-test-parameter-pipeline-c877j"
                },
            },
            "task_results_completion_status": {
                "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-1412890278": True,
                "pipeline-test-parameter-pipeline-c877j-flow-tfgmn-4267990770": True,
            },
        },
    )


@pytest.fixture
def test_hera_torch_gpu_workflow_model(test_datetime):

    return WorkflowModel(
        metadata={
            "annotations": {
                "karpenter.sh/do-not-disrupt": "true",
                "workflows.argoproj.io/pod-name-format": "v2",
            },
            "creation_timestamp": test_datetime,
            "generate_name": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-",
            "generation": 13,
            "labels": {
                "bettmensch.ai/pipeline-id": "612226a1-b40f-4f68-92c3-ea8a5d6b3995",
                "bettmensch.ai/pipeline-name": "pipeline-test-torch-gpu-pipeline-7c4zp",
                "workflows.argoproj.io/completed": "true",
                "workflows.argoproj.io/creator": "system-serviceaccount-argo-argo-server",
                "workflows.argoproj.io/phase": "Succeeded",
            },
            "managed_fields": [
                {
                    "api_version": "argoproj.io/v1alpha1",
                    "fields_type": "FieldsV1",
                    "fields_v1": {},
                    "manager": "argo",
                    "operation": "Update",
                    "time": test_datetime,
                },
                {
                    "api_version": "argoproj.io/v1alpha1",
                    "fields_type": "FieldsV1",
                    "fields_v1": {},
                    "manager": "workflow-controller",
                    "operation": "Update",
                    "time": test_datetime,
                },
            ],
            "name": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf",
            "namespace": "argo",
            "resource_version": "11463",
            "uid": "ae69b1e3-a235-44d5-8667-bef63fc15821",
        },
        spec={
            "arguments": {
                "parameters": [
                    {"name": "n_iter", "value": "15"},
                    {"name": "n_seconds_sleep", "value": "2"},
                ]
            },
            "workflow_template_ref": {
                "name": "pipeline-test-torch-gpu-pipeline-7c4zp"
            },
        },
        status={
            "artifact_gc_status": {"not_specified": True},
            "artifact_repository_ref": {
                "artifact_repository": {
                    "s3": {
                        "bucket": "bettmensch-ai-artifact-repository",
                        "endpoint": "s3.us-east-2.amazonaws.com",
                        "insecure": True,
                        "key_format": "argo-workflows/{{workflow.name}}/{{pod.name}}",
                    }
                },
                "config_map": "artifact-repositories",
                "key": "bettmensch-ai-artifact-repository",
                "namespace": "argo",
            },
            "conditions": [
                {"status": "False", "type": "PodRunning"},
                {"status": "True", "type": "Completed"},
            ],
            "finished_at": test_datetime,
            "nodes": {
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf": {
                    "children": [
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-414716060"
                    ],
                    "display_name": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf",
                    "finished_at": test_datetime,
                    "id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf",
                    "name": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf",
                    "outbound_nodes": [
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-947069694",
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-41628430",
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-1368447231",
                    ],
                    "phase": "Succeeded",
                    "progress": "5/5",
                    "resources_duration": {
                        "cpu": 23,
                        "memory": 1644,
                        "nvidia.com/gpu": 190,
                    },
                    "started_at": test_datetime,
                    "template_name": "bettmensch-ai-outer-dag",
                    "template_scope": "local/",
                    "type": "DAG",
                },
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-1368447231": {
                    "boundary_id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-414716060",
                    "display_name": "torch-ddp-delete-torch-ddp-service",
                    "finished_at": test_datetime,
                    "host_node_name": "ip-10-0-48-85.us-east-2.compute.internal",
                    "id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-1368447231",
                    "name": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf.bettmensch-ai-inner-dag.torch-ddp-delete-torch-ddp-service",
                    "outputs": {"exit_code": "0"},
                    "phase": "Succeeded",
                    "progress": "1/1",
                    "resources_duration": {"cpu": 0, "memory": 0},
                    "started_at": test_datetime,
                    "template_name": "torch-ddp-delete-torch-ddp-service",
                    "template_scope": "local/",
                    "type": "Pod",
                },
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-1861925387": {
                    "boundary_id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-414716060",
                    "children": [
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-2733896051",
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-1368447231",
                    ],
                    "display_name": "torch-ddp-0(0)",
                    "finished_at": test_datetime,
                    "host_node_name": "ip-10-0-50-210.us-east-2.compute.internal",
                    "id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-1861925387",
                    "inputs": {
                        "parameters": [
                            {
                                "default": "100",
                                "name": "n_iter",
                                "value": "15",
                            },
                            {
                                "default": "10",
                                "name": "n_seconds_sleep",
                                "value": "2",
                            },
                            {
                                "default": "null",
                                "name": "duration",
                                "value": "null",
                            },
                        ]
                    },
                    "name": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf.bettmensch-ai-inner-dag.torch-ddp-0(0)",
                    "node_flag": {"retried": True},
                    "outputs": {
                        "exit_code": "0",
                        "parameters": [
                            {
                                "name": "duration",
                                "value": "30",
                                "value_from": {"path": "duration"},
                            }
                        ],
                    },
                    "phase": "Succeeded",
                    "progress": "1/1",
                    "resources_duration": {
                        "cpu": 11,
                        "memory": 839,
                        "nvidia.com/gpu": 99,
                    },
                    "started_at": test_datetime,
                    "template_name": "torch-ddp-0",
                    "template_scope": "local/",
                    "type": "Pod",
                },
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-2020597252": {
                    "boundary_id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-414716060",
                    "children": [
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-47634872",
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-4097461059",
                    ],
                    "display_name": "torch-ddp-create-torch-ddp-service",
                    "finished_at": test_datetime,
                    "host_node_name": "ip-10-0-49-235.us-east-2.compute.internal",
                    "id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-2020597252",
                    "name": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf.bettmensch-ai-inner-dag.torch-ddp-create-torch-ddp-service",
                    "outputs": {"exit_code": "0"},
                    "phase": "Succeeded",
                    "progress": "1/1",
                    "resources_duration": {"cpu": 0, "memory": 1},
                    "started_at": test_datetime,
                    "template_name": "torch-ddp-create-torch-ddp-service",
                    "template_scope": "local/",
                    "type": "Pod",
                },
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-2733896051": {
                    "boundary_id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-414716060",
                    "children": [
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-947069694"
                    ],
                    "display_name": "show-duration-param-0",
                    "finished_at": test_datetime,
                    "id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-2733896051",
                    "inputs": {"parameters": [{"name": "a", "value": "30"}]},
                    "name": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf.bettmensch-ai-inner-dag.show-duration-param-0",
                    "outputs": {"exit_code": "0"},
                    "phase": "Succeeded",
                    "progress": "1/1",
                    "resources_duration": {"cpu": 1, "memory": 27},
                    "started_at": test_datetime,
                    "template_name": "show-duration-param",
                    "template_scope": "local/",
                    "type": "Retry",
                },
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-4097461059": {
                    "boundary_id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-414716060",
                    "children": [
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-41628430"
                    ],
                    "display_name": "torch-ddp-0-worker-1",
                    "finished_at": test_datetime,
                    "id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-4097461059",
                    "inputs": {
                        "parameters": [
                            {
                                "default": "100",
                                "name": "n_iter",
                                "value": "15",
                            },
                            {
                                "default": "10",
                                "name": "n_seconds_sleep",
                                "value": "2",
                            },
                            {
                                "default": "null",
                                "name": "duration",
                                "value": "null",
                            },
                        ]
                    },
                    "name": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf.bettmensch-ai-inner-dag.torch-ddp-0-worker-1",
                    "outputs": {
                        "exit_code": "0",
                        "parameters": [
                            {
                                "name": "duration",
                                "value": "30",
                                "value_from": {"path": "duration"},
                            }
                        ],
                    },
                    "phase": "Succeeded",
                    "progress": "1/1",
                    "resources_duration": {
                        "cpu": 11,
                        "memory": 777,
                        "nvidia.com/gpu": 91,
                    },
                    "started_at": test_datetime,
                    "template_name": "torch-ddp-1",
                    "template_scope": "local/",
                    "type": "Retry",
                },
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-414716060": {
                    "boundary_id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf",
                    "children": [
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-2020597252"
                    ],
                    "display_name": "bettmensch-ai-inner-dag",
                    "finished_at": test_datetime,
                    "id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-414716060",
                    "inputs": {
                        "parameters": [
                            {"name": "n_iter", "value": "15"},
                            {"name": "n_seconds_sleep", "value": "2"},
                        ]
                    },
                    "name": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf.bettmensch-ai-inner-dag",
                    "outbound_nodes": [
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-947069694",
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-41628430",
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-1368447231",
                    ],
                    "phase": "Succeeded",
                    "progress": "5/5",
                    "resources_duration": {
                        "cpu": 23,
                        "memory": 1644,
                        "nvidia.com/gpu": 190,
                    },
                    "started_at": test_datetime,
                    "template_name": "bettmensch-ai-inner-dag",
                    "template_scope": "local/",
                    "type": "DAG",
                },
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-41628430": {
                    "boundary_id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-414716060",
                    "display_name": "torch-ddp-0-worker-1(0)",
                    "finished_at": test_datetime,
                    "host_node_name": "ip-10-0-50-218.us-east-2.compute.internal",
                    "id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-41628430",
                    "inputs": {
                        "parameters": [
                            {
                                "default": "100",
                                "name": "n_iter",
                                "value": "15",
                            },
                            {
                                "default": "10",
                                "name": "n_seconds_sleep",
                                "value": "2",
                            },
                            {
                                "default": "null",
                                "name": "duration",
                                "value": "null",
                            },
                        ]
                    },
                    "name": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf.bettmensch-ai-inner-dag.torch-ddp-0-worker-1(0)",
                    "node_flag": {"retried": True},
                    "outputs": {
                        "exit_code": "0",
                        "parameters": [
                            {
                                "name": "duration",
                                "value": "30",
                                "value_from": {"path": "duration"},
                            }
                        ],
                    },
                    "phase": "Succeeded",
                    "progress": "1/1",
                    "resources_duration": {
                        "cpu": 11,
                        "memory": 777,
                        "nvidia.com/gpu": 91,
                    },
                    "started_at": test_datetime,
                    "template_name": "torch-ddp-1",
                    "template_scope": "local/",
                    "type": "Pod",
                },
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-47634872": {
                    "boundary_id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-414716060",
                    "children": [
                        "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-1861925387"
                    ],
                    "display_name": "torch-ddp-0",
                    "finished_at": test_datetime,
                    "id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-47634872",
                    "inputs": {
                        "parameters": [
                            {
                                "default": "100",
                                "name": "n_iter",
                                "value": "15",
                            },
                            {
                                "default": "10",
                                "name": "n_seconds_sleep",
                                "value": "2",
                            },
                            {
                                "default": "null",
                                "name": "duration",
                                "value": "null",
                            },
                        ]
                    },
                    "name": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf.bettmensch-ai-inner-dag.torch-ddp-0",
                    "outputs": {
                        "exit_code": "0",
                        "parameters": [
                            {
                                "name": "duration",
                                "value": "30",
                                "value_from": {"path": "duration"},
                            }
                        ],
                    },
                    "phase": "Succeeded",
                    "progress": "3/3",
                    "resources_duration": {
                        "cpu": 12,
                        "memory": 866,
                        "nvidia.com/gpu": 99,
                    },
                    "started_at": test_datetime,
                    "template_name": "torch-ddp-0",
                    "template_scope": "local/",
                    "type": "Retry",
                },
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-947069694": {
                    "boundary_id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-414716060",
                    "display_name": "show-duration-param-0(0)",
                    "finished_at": test_datetime,
                    "host_node_name": "ip-10-0-49-235.us-east-2.compute.internal",
                    "id": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-947069694",
                    "inputs": {"parameters": [{"name": "a", "value": "30"}]},
                    "name": "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf.bettmensch-ai-inner-dag.show-duration-param-0(0)",
                    "node_flag": {"retried": True},
                    "outputs": {"exit_code": "0"},
                    "phase": "Succeeded",
                    "progress": "1/1",
                    "resources_duration": {"cpu": 1, "memory": 27},
                    "started_at": test_datetime,
                    "template_name": "show-duration-param",
                    "template_scope": "local/",
                    "type": "Pod",
                },
            },
            "phase": "Succeeded",
            "progress": "5/5",
            "resources_duration": {
                "cpu": 23,
                "memory": 1644,
                "nvidia.com/gpu": 190,
            },
            "started_at": test_datetime,
            "stored_templates": {
                "namespaced/pipeline-test-torch-gpu-pipeline-7c4zp/bettmensch-ai-inner-dag": {
                    "dag": {
                        "tasks": [
                            {
                                "arguments": {},
                                "name": "torch-ddp-create-torch-ddp-service",
                                "template": "torch-ddp-create-torch-ddp-service",
                            },
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "n_iter",
                                            "value": "{{inputs.parameters.n_iter}}",
                                        },
                                        {
                                            "name": "n_seconds_sleep",
                                            "value": "{{inputs.parameters.n_seconds_sleep}}",
                                        },
                                    ]
                                },
                                "depends": "torch-ddp-create-torch-ddp-service",
                                "name": "torch-ddp-0",
                                "template": "torch-ddp-0",
                            },
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "n_iter",
                                            "value": "{{inputs.parameters.n_iter}}",
                                        },
                                        {
                                            "name": "n_seconds_sleep",
                                            "value": "{{inputs.parameters.n_seconds_sleep}}",
                                        },
                                    ]
                                },
                                "depends": "torch-ddp-create-torch-ddp-service",
                                "name": "torch-ddp-0-worker-1",
                                "template": "torch-ddp-1",
                            },
                            {
                                "arguments": {},
                                "depends": "torch-ddp-0",
                                "name": "torch-ddp-delete-torch-ddp-service",
                                "template": "torch-ddp-delete-torch-ddp-service",
                            },
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "a",
                                            "value": "{{tasks.torch-ddp-0.outputs.parameters.duration}}",
                                        }
                                    ]
                                },
                                "depends": "torch-ddp-0",
                                "name": "show-duration-param-0",
                                "template": "show-duration-param",
                            },
                        ]
                    },
                    "inputs": {
                        "parameters": [
                            {"name": "n_iter"},
                            {"name": "n_seconds_sleep"},
                        ]
                    },
                    "metadata": {},
                    "name": "bettmensch-ai-inner-dag",
                    "outputs": {},
                },
                "namespaced/pipeline-test-torch-gpu-pipeline-7c4zp/bettmensch-ai-outer-dag": {
                    "dag": {
                        "tasks": [
                            {
                                "arguments": {
                                    "parameters": [
                                        {
                                            "name": "n_iter",
                                            "value": "{{workflow.parameters.n_iter}}",
                                        },
                                        {
                                            "name": "n_seconds_sleep",
                                            "value": "{{workflow.parameters.n_seconds_sleep}}",
                                        },
                                    ]
                                },
                                "name": "bettmensch-ai-inner-dag",
                                "template": "bettmensch-ai-inner-dag",
                            }
                        ]
                    },
                    "inputs": {},
                    "metadata": {},
                    "name": "bettmensch-ai-outer-dag",
                    "outputs": {},
                },
                "namespaced/pipeline-test-torch-gpu-pipeline-7c4zp/show-duration-param": {
                    "inputs": {"parameters": [{"name": "a"}]},
                    "metadata": {},
                    "name": "show-duration-param",
                    "outputs": {},
                    "retry_strategy": {
                        "limit": "1",
                        "retry_policy": "OnError",
                    },
                    "script": {
                        "command": ["python"],
                        "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                        "image_pull_policy": "Always",
                        "name": "",
                        "resources": {
                            "limits": {"cpu": "100m", "memory": "100Mi"},
                            "requests": {"cpu": "100m", "memory": "100Mi"},
                        },
                        "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: a = json.loads(r'''{{inputs.parameters.a}}''')\nexcept: a = r'''{{inputs.parameters.a}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\ndef show_parameter(a: InputParameter) -> None:\n    \"\"\"When decorated with the bettmensch_ai.components.component decorator,\n    implements a bettmensch_ai.Component that prints the values of its\n    InputParameter.\"\"\"\n    print(f'Content of input parameter a is: {a}')\n\nshow_parameter(a)\n",
                    },
                },
                "namespaced/pipeline-test-torch-gpu-pipeline-7c4zp/torch-ddp-0": {
                    "inputs": {
                        "parameters": [
                            {"default": "100", "name": "n_iter"},
                            {"default": "10", "name": "n_seconds_sleep"},
                            {"default": "null", "name": "duration"},
                        ]
                    },
                    "metadata": {
                        "labels": {
                            "torch-job": "torch-ddp-0",
                            "torch-node": "0",
                        }
                    },
                    "name": "torch-ddp-0",
                    "outputs": {
                        "parameters": [
                            {
                                "name": "duration",
                                "value_from": {"path": "duration"},
                            }
                        ]
                    },
                    "pod_spec_patch": "topologySpreadConstraints:\n- maxSkew: 1\n  topologyKey: kubernetes.io/hostname\n  whenUnsatisfiable: DoNotSchedule\n  labelSelector:\n    matchExpressions:\n      - { key: torch-node, operator: In, values: ['0','1','2','3','4','5']}",
                    "retry_strategy": {
                        "limit": "1",
                        "retry_policy": "OnError",
                    },
                    "script": {
                        "command": ["python"],
                        "env": [
                            {"name": "NCCL_DEBUG", "value": "INFO"},
                            {
                                "name": "bettmensch_ai_torch_ddp_min_nodes",
                                "value": "2",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_max_nodes",
                                "value": "2",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_node_rank",
                                "value": "0",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_nproc_per_node",
                                "value": "1",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_max_restarts",
                                "value": "1",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_start_method",
                                "value": "fork",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_rdzv_backend",
                                "value": "static",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_rdzv_endpoint_url",
                                "value": "torch-ddp-0-{{workflow.uid}}.argo.svc.cluster.local",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_rdzv_endpoint_port",
                                "value": "29200",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_run_id",
                                "value": "1",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_tee",
                                "value": "0",
                            },
                        ],
                        "image": "bettmensch88/bettmensch.ai-pytorch:3.11-latest",
                        "image_pull_policy": "Always",
                        "name": "",
                        "ports": [
                            {
                                "container_port": 29200,
                                "name": "ddp",
                                "protocol": "TCP",
                            }
                        ],
                        "resources": {
                            "limits": {
                                "cpu": "100m",
                                "memory": "700Mi",
                                "nvidia.com/gpu": "1",
                            },
                            "requests": {
                                "cpu": "100m",
                                "memory": "700Mi",
                                "nvidia.com/gpu": "1",
                            },
                        },
                        "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: n_iter = json.loads(r'''{{inputs.parameters.n_iter}}''')\nexcept: n_iter = r'''{{inputs.parameters.n_iter}}'''\ntry: n_seconds_sleep = json.loads(r'''{{inputs.parameters.n_seconds_sleep}}''')\nexcept: n_seconds_sleep = r'''{{inputs.parameters.n_seconds_sleep}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputParameter\nduration = OutputParameter(\"duration\")\n\ndef tensor_reduce(n_iter: InputParameter=100, n_seconds_sleep: InputParameter=10, duration: OutputParameter=None) -> None:\n    \"\"\"When decorated with the torch_component decorator, implements a\n    bettmensch_ai.TorchComponent that runs a torch DDP across pods and nodes in\n    your K8s cluster.\"\"\"\n    import time\n    from datetime import datetime as dt\n    import GPUtil\n    import torch\n    import torch.distributed as dist\n    from bettmensch_ai.pipelines.component.torch_ddp import LaunchContext\n    has_gpu = torch.cuda.is_available()\n    ddp_context = LaunchContext()\n    print(f'GPU present: {has_gpu}')\n    if has_gpu:\n        dist.init_process_group(backend='nccl')\n    else:\n        dist.init_process_group(backend='gloo')\n    for i in range(1, n_iter + 1):\n        time.sleep(n_seconds_sleep)\n        GPUtil.showUtilization()\n        a = torch.tensor([ddp_context.rank])\n        print(f'{i}/{n_iter}: @{dt.now()}')\n        print(f'{i}/{n_iter}: Backend {dist.get_backend()}')\n        print(f'{i}/{n_iter}: Global world size: {ddp_context.world_size}')\n        print(f'{i}/{n_iter}: Global worker process rank: {ddp_context.rank}')\n        print(f'{i}/{n_iter}: This makes me worker process {ddp_context.rank + 1}/{ddp_context.world_size} globally!')\n        print(f'{i}/{n_iter}: Local rank of worker: {ddp_context.local_rank}')\n        print(f'{i}/{n_iter}: Local world size: {ddp_context.local_world_size}')\n        print(f'{i}/{n_iter}: This makes me worker process {ddp_context.local_rank + 1}/{ddp_context.local_world_size} locally!')\n        print(f'{i}/{n_iter}: Node/pod rank: {ddp_context.group_rank}')\n        if has_gpu:\n            device = torch.device(f'cuda:{ddp_context.local_rank}')\n            device_count = torch.cuda.device_count()\n            print(f'{i}/{n_iter}: GPU count: {device_count}')\n            device_name = torch.cuda.get_device_name(ddp_context.local_rank)\n            print(f'{i}/{n_iter}: GPU name: {device_name}')\n            device_property = torch.cuda.get_device_capability(device)\n            print(f'{i}/{n_iter}: GPU property: {device_property}')\n        else:\n            device = torch.device('cpu')\n        a_placed = a.to(device)\n        print(f'{i}/{n_iter}: Pre-`all_reduce` tensor: {a_placed}')\n        dist.all_reduce(a_placed)\n        print(f'{i}/{n_iter}: Post-`all_reduce` tensor: {a_placed}')\n        print('===================================================')\n    if duration is not None:\n        duration_seconds = n_iter * n_seconds_sleep\n        duration.assign(duration_seconds)\n\nfrom torch.distributed.elastic.multiprocessing.errors import record\n\ntensor_reduce=record(tensor_reduce)\n\nfrom bettmensch_ai.pipelines.component import as_torch_ddp\n\ntorch_ddp_decorator=as_torch_ddp()\n\ntorch_ddp_function=torch_ddp_decorator(tensor_reduce)\n\n\ntorch_ddp_function(n_iter,n_seconds_sleep,duration)",  # noqa: E501
                    },
                    "tolerations": [
                        {
                            "effect": "NoSchedule",
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                        }
                    ],
                },
                "namespaced/pipeline-test-torch-gpu-pipeline-7c4zp/torch-ddp-1": {
                    "inputs": {
                        "parameters": [
                            {"default": "100", "name": "n_iter"},
                            {"default": "10", "name": "n_seconds_sleep"},
                            {"default": "null", "name": "duration"},
                        ]
                    },
                    "metadata": {
                        "labels": {
                            "torch-job": "torch-ddp-0",
                            "torch-node": "1",
                        }
                    },
                    "name": "torch-ddp-1",
                    "outputs": {
                        "parameters": [
                            {
                                "name": "duration",
                                "value_from": {"path": "duration"},
                            }
                        ]
                    },
                    "pod_spec_patch": "topologySpreadConstraints:\n- maxSkew: 1\n  topologyKey: kubernetes.io/hostname\n  whenUnsatisfiable: DoNotSchedule\n  labelSelector:\n    matchExpressions:\n      - { key: torch-node, operator: In, values: ['0','1','2','3','4','5']}",
                    "retry_strategy": {
                        "limit": "1",
                        "retry_policy": "OnError",
                    },
                    "script": {
                        "command": ["python"],
                        "env": [
                            {"name": "NCCL_DEBUG", "value": "INFO"},
                            {
                                "name": "bettmensch_ai_torch_ddp_min_nodes",
                                "value": "2",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_max_nodes",
                                "value": "2",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_node_rank",
                                "value": "1",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_nproc_per_node",
                                "value": "1",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_max_restarts",
                                "value": "1",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_start_method",
                                "value": "fork",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_rdzv_backend",
                                "value": "static",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_rdzv_endpoint_url",
                                "value": "torch-ddp-0-{{workflow.uid}}.argo.svc.cluster.local",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_rdzv_endpoint_port",
                                "value": "29200",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_run_id",
                                "value": "1",
                            },
                            {
                                "name": "bettmensch_ai_torch_ddp_tee",
                                "value": "0",
                            },
                        ],
                        "image": "bettmensch88/bettmensch.ai-pytorch:3.11-latest",
                        "image_pull_policy": "Always",
                        "name": "",
                        "resources": {
                            "limits": {
                                "cpu": "100m",
                                "memory": "700Mi",
                                "nvidia.com/gpu": "1",
                            },
                            "requests": {
                                "cpu": "100m",
                                "memory": "700Mi",
                                "nvidia.com/gpu": "1",
                            },
                        },
                        "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: n_iter = json.loads(r'''{{inputs.parameters.n_iter}}''')\nexcept: n_iter = r'''{{inputs.parameters.n_iter}}'''\ntry: n_seconds_sleep = json.loads(r'''{{inputs.parameters.n_seconds_sleep}}''')\nexcept: n_seconds_sleep = r'''{{inputs.parameters.n_seconds_sleep}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputParameter\nduration = OutputParameter(\"duration\")\n\ndef tensor_reduce(n_iter: InputParameter=100, n_seconds_sleep: InputParameter=10, duration: OutputParameter=None) -> None:\n    \"\"\"When decorated with the torch_component decorator, implements a\n    bettmensch_ai.TorchComponent that runs a torch DDP across pods and nodes in\n    your K8s cluster.\"\"\"\n    import time\n    from datetime import datetime as dt\n    import GPUtil\n    import torch\n    import torch.distributed as dist\n    from bettmensch_ai.pipelines.component.torch_ddp import LaunchContext\n    has_gpu = torch.cuda.is_available()\n    ddp_context = LaunchContext()\n    print(f'GPU present: {has_gpu}')\n    if has_gpu:\n        dist.init_process_group(backend='nccl')\n    else:\n        dist.init_process_group(backend='gloo')\n    for i in range(1, n_iter + 1):\n        time.sleep(n_seconds_sleep)\n        GPUtil.showUtilization()\n        a = torch.tensor([ddp_context.rank])\n        print(f'{i}/{n_iter}: @{dt.now()}')\n        print(f'{i}/{n_iter}: Backend {dist.get_backend()}')\n        print(f'{i}/{n_iter}: Global world size: {ddp_context.world_size}')\n        print(f'{i}/{n_iter}: Global worker process rank: {ddp_context.rank}')\n        print(f'{i}/{n_iter}: This makes me worker process {ddp_context.rank + 1}/{ddp_context.world_size} globally!')\n        print(f'{i}/{n_iter}: Local rank of worker: {ddp_context.local_rank}')\n        print(f'{i}/{n_iter}: Local world size: {ddp_context.local_world_size}')\n        print(f'{i}/{n_iter}: This makes me worker process {ddp_context.local_rank + 1}/{ddp_context.local_world_size} locally!')\n        print(f'{i}/{n_iter}: Node/pod rank: {ddp_context.group_rank}')\n        if has_gpu:\n            device = torch.device(f'cuda:{ddp_context.local_rank}')\n            device_count = torch.cuda.device_count()\n            print(f'{i}/{n_iter}: GPU count: {device_count}')\n            device_name = torch.cuda.get_device_name(ddp_context.local_rank)\n            print(f'{i}/{n_iter}: GPU name: {device_name}')\n            device_property = torch.cuda.get_device_capability(device)\n            print(f'{i}/{n_iter}: GPU property: {device_property}')\n        else:\n            device = torch.device('cpu')\n        a_placed = a.to(device)\n        print(f'{i}/{n_iter}: Pre-`all_reduce` tensor: {a_placed}')\n        dist.all_reduce(a_placed)\n        print(f'{i}/{n_iter}: Post-`all_reduce` tensor: {a_placed}')\n        print('===================================================')\n    if duration is not None:\n        duration_seconds = n_iter * n_seconds_sleep\n        duration.assign(duration_seconds)\n\nfrom torch.distributed.elastic.multiprocessing.errors import record\n\ntensor_reduce=record(tensor_reduce)\n\nfrom bettmensch_ai.pipelines.component import as_torch_ddp\n\ntorch_ddp_decorator=as_torch_ddp()\n\ntorch_ddp_function=torch_ddp_decorator(tensor_reduce)\n\n\ntorch_ddp_function(n_iter,n_seconds_sleep,duration)",
                    },
                    "tolerations": [
                        {
                            "effect": "NoSchedule",
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                        }
                    ],
                },
                "namespaced/pipeline-test-torch-gpu-pipeline-7c4zp/torch-ddp-create-torch-ddp-service": {
                    "inputs": {},
                    "metadata": {},
                    "name": "torch-ddp-create-torch-ddp-service",
                    "outputs": {},
                    "resource": {
                        "action": "create",
                        "manifest": "apiVersion: v1\nkind: Service\nmetadata:\n  name: torch-ddp-0-{{workflow.uid}}\n  namespace: argo\n  labels:\n    workflows.argoproj.io/workflow: {{workflow.name}}\n    torch-job: torch-ddp-0\nspec:\n  clusterIP: None  # ClusterIP set to None for headless service.\n  ports:\n  - name: ddp  # Port for torchrun master<->worker node coms.\n    port: 29200\n    targetPort: 29200\n  selector:\n    workflows.argoproj.io/workflow: {{workflow.name}}\n    torch-job: torch-ddp-0\n    torch-node: '0'  # Selector for pods associated with this service.\n",
                    },
                },
                "namespaced/pipeline-test-torch-gpu-pipeline-7c4zp/torch-ddp-delete-torch-ddp-service": {
                    "inputs": {},
                    "metadata": {},
                    "name": "torch-ddp-delete-torch-ddp-service",
                    "outputs": {},
                    "resource": {
                        "action": "delete",
                        "flags": [
                            "service",
                            "--selector",
                            "torch-job=torch-ddp-0,workflows.argoproj.io/workflow={{workflow.name}}",
                            "-n",
                            "argo",
                        ],
                    },
                },
            },
            "stored_workflow_template_spec": {
                "arguments": {
                    "parameters": [
                        {"name": "n_iter", "value": "15"},
                        {"name": "n_seconds_sleep", "value": "2"},
                    ]
                },
                "entrypoint": "bettmensch-ai-outer-dag",
                "service_account_name": "argo-workflow",
                "templates": [
                    {
                        "inputs": {},
                        "metadata": {},
                        "name": "torch-ddp-create-torch-ddp-service",
                        "outputs": {},
                        "resource": {
                            "action": "create",
                            "manifest": "apiVersion: v1\nkind: Service\nmetadata:\n  name: torch-ddp-0-{{workflow.uid}}\n  namespace: argo\n  labels:\n    workflows.argoproj.io/workflow: {{workflow.name}}\n    torch-job: torch-ddp-0\nspec:\n  clusterIP: None  # ClusterIP set to None for headless service.\n  ports:\n  - name: ddp  # Port for torchrun master<->worker node coms.\n    port: 29200\n    targetPort: 29200\n  selector:\n    workflows.argoproj.io/workflow: {{workflow.name}}\n    torch-job: torch-ddp-0\n    torch-node: '0'  # Selector for pods associated with this service.\n",
                        },
                    },
                    {
                        "inputs": {},
                        "metadata": {},
                        "name": "torch-ddp-delete-torch-ddp-service",
                        "outputs": {},
                        "resource": {
                            "action": "delete",
                            "flags": [
                                "service",
                                "--selector",
                                "torch-job=torch-ddp-0,workflows.argoproj.io/workflow={{workflow.name}}",
                                "-n",
                                "argo",
                            ],
                        },
                    },
                    {
                        "dag": {
                            "tasks": [
                                {
                                    "arguments": {},
                                    "name": "torch-ddp-create-torch-ddp-service",
                                    "template": "torch-ddp-create-torch-ddp-service",
                                },
                                {
                                    "arguments": {
                                        "parameters": [
                                            {
                                                "name": "n_iter",
                                                "value": "{{inputs.parameters.n_iter}}",
                                            },
                                            {
                                                "name": "n_seconds_sleep",
                                                "value": "{{inputs.parameters.n_seconds_sleep}}",
                                            },
                                        ]
                                    },
                                    "depends": "torch-ddp-create-torch-ddp-service",
                                    "name": "torch-ddp-0",
                                    "template": "torch-ddp-0",
                                },
                                {
                                    "arguments": {
                                        "parameters": [
                                            {
                                                "name": "n_iter",
                                                "value": "{{inputs.parameters.n_iter}}",
                                            },
                                            {
                                                "name": "n_seconds_sleep",
                                                "value": "{{inputs.parameters.n_seconds_sleep}}",
                                            },
                                        ]
                                    },
                                    "depends": "torch-ddp-create-torch-ddp-service",
                                    "name": "torch-ddp-0-worker-1",
                                    "template": "torch-ddp-1",
                                },
                                {
                                    "arguments": {},
                                    "depends": "torch-ddp-0",
                                    "name": "torch-ddp-delete-torch-ddp-service",
                                    "template": "torch-ddp-delete-torch-ddp-service",
                                },
                                {
                                    "arguments": {
                                        "parameters": [
                                            {
                                                "name": "a",
                                                "value": "{{tasks.torch-ddp-0.outputs.parameters.duration}}",
                                            }
                                        ]
                                    },
                                    "depends": "torch-ddp-0",
                                    "name": "show-duration-param-0",
                                    "template": "show-duration-param",
                                },
                            ]
                        },
                        "inputs": {
                            "parameters": [
                                {"name": "n_iter"},
                                {"name": "n_seconds_sleep"},
                            ]
                        },
                        "metadata": {},
                        "name": "bettmensch-ai-inner-dag",
                        "outputs": {},
                    },
                    {
                        "inputs": {
                            "parameters": [
                                {"default": "100", "name": "n_iter"},
                                {"default": "10", "name": "n_seconds_sleep"},
                                {"default": "null", "name": "duration"},
                            ]
                        },
                        "metadata": {
                            "labels": {
                                "torch-job": "torch-ddp-0",
                                "torch-node": "0",
                            }
                        },
                        "name": "torch-ddp-0",
                        "outputs": {
                            "parameters": [
                                {
                                    "name": "duration",
                                    "value_from": {"path": "duration"},
                                }
                            ]
                        },
                        "pod_spec_patch": "topologySpreadConstraints:\n- maxSkew: 1\n  topologyKey: kubernetes.io/hostname\n  whenUnsatisfiable: DoNotSchedule\n  labelSelector:\n    matchExpressions:\n      - { key: torch-node, operator: In, values: ['0','1','2','3','4','5']}",
                        "retry_strategy": {
                            "limit": "1",
                            "retry_policy": "OnError",
                        },
                        "script": {
                            "command": ["python"],
                            "env": [
                                {"name": "NCCL_DEBUG", "value": "INFO"},
                                {
                                    "name": "bettmensch_ai_torch_ddp_min_nodes",
                                    "value": "2",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_max_nodes",
                                    "value": "2",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_node_rank",
                                    "value": "0",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_nproc_per_node",
                                    "value": "1",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_max_restarts",
                                    "value": "1",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_start_method",
                                    "value": "fork",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_rdzv_backend",
                                    "value": "static",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_rdzv_endpoint_url",
                                    "value": "torch-ddp-0-{{workflow.uid}}.argo.svc.cluster.local",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_rdzv_endpoint_port",
                                    "value": "29200",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_run_id",
                                    "value": "1",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_tee",
                                    "value": "0",
                                },
                            ],
                            "image": "bettmensch88/bettmensch.ai-pytorch:3.11-latest",
                            "image_pull_policy": "Always",
                            "name": "",
                            "ports": [
                                {
                                    "container_port": 29200,
                                    "name": "ddp",
                                    "protocol": "TCP",
                                }
                            ],
                            "resources": {
                                "limits": {
                                    "cpu": "100m",
                                    "memory": "700Mi",
                                    "nvidia.com/gpu": "1",
                                },
                                "requests": {
                                    "cpu": "100m",
                                    "memory": "700Mi",
                                    "nvidia.com/gpu": "1",
                                },
                            },
                            "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: n_iter = json.loads(r'''{{inputs.parameters.n_iter}}''')\nexcept: n_iter = r'''{{inputs.parameters.n_iter}}'''\ntry: n_seconds_sleep = json.loads(r'''{{inputs.parameters.n_seconds_sleep}}''')\nexcept: n_seconds_sleep = r'''{{inputs.parameters.n_seconds_sleep}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputParameter\nduration = OutputParameter(\"duration\")\n\ndef tensor_reduce(n_iter: InputParameter=100, n_seconds_sleep: InputParameter=10, duration: OutputParameter=None) -> None:\n    \"\"\"When decorated with the torch_component decorator, implements a\n    bettmensch_ai.TorchComponent that runs a torch DDP across pods and nodes in\n    your K8s cluster.\"\"\"\n    import time\n    from datetime import datetime as dt\n    import GPUtil\n    import torch\n    import torch.distributed as dist\n    from bettmensch_ai.pipelines.component.torch_ddp import LaunchContext\n    has_gpu = torch.cuda.is_available()\n    ddp_context = LaunchContext()\n    print(f'GPU present: {has_gpu}')\n    if has_gpu:\n        dist.init_process_group(backend='nccl')\n    else:\n        dist.init_process_group(backend='gloo')\n    for i in range(1, n_iter + 1):\n        time.sleep(n_seconds_sleep)\n        GPUtil.showUtilization()\n        a = torch.tensor([ddp_context.rank])\n        print(f'{i}/{n_iter}: @{dt.now()}')\n        print(f'{i}/{n_iter}: Backend {dist.get_backend()}')\n        print(f'{i}/{n_iter}: Global world size: {ddp_context.world_size}')\n        print(f'{i}/{n_iter}: Global worker process rank: {ddp_context.rank}')\n        print(f'{i}/{n_iter}: This makes me worker process {ddp_context.rank + 1}/{ddp_context.world_size} globally!')\n        print(f'{i}/{n_iter}: Local rank of worker: {ddp_context.local_rank}')\n        print(f'{i}/{n_iter}: Local world size: {ddp_context.local_world_size}')\n        print(f'{i}/{n_iter}: This makes me worker process {ddp_context.local_rank + 1}/{ddp_context.local_world_size} locally!')\n        print(f'{i}/{n_iter}: Node/pod rank: {ddp_context.group_rank}')\n        if has_gpu:\n            device = torch.device(f'cuda:{ddp_context.local_rank}')\n            device_count = torch.cuda.device_count()\n            print(f'{i}/{n_iter}: GPU count: {device_count}')\n            device_name = torch.cuda.get_device_name(ddp_context.local_rank)\n            print(f'{i}/{n_iter}: GPU name: {device_name}')\n            device_property = torch.cuda.get_device_capability(device)\n            print(f'{i}/{n_iter}: GPU property: {device_property}')\n        else:\n            device = torch.device('cpu')\n        a_placed = a.to(device)\n        print(f'{i}/{n_iter}: Pre-`all_reduce` tensor: {a_placed}')\n        dist.all_reduce(a_placed)\n        print(f'{i}/{n_iter}: Post-`all_reduce` tensor: {a_placed}')\n        print('===================================================')\n    if duration is not None:\n        duration_seconds = n_iter * n_seconds_sleep\n        duration.assign(duration_seconds)\n\nfrom torch.distributed.elastic.multiprocessing.errors import record\n\ntensor_reduce=record(tensor_reduce)\n\nfrom bettmensch_ai.pipelines.component import as_torch_ddp\n\ntorch_ddp_decorator=as_torch_ddp()\n\ntorch_ddp_function=torch_ddp_decorator(tensor_reduce)\n\n\ntorch_ddp_function(n_iter,n_seconds_sleep,duration)",
                        },
                        "tolerations": [
                            {
                                "effect": "NoSchedule",
                                "key": "nvidia.com/gpu",
                                "operator": "Exists",
                            }
                        ],
                    },
                    {
                        "inputs": {
                            "parameters": [
                                {"default": "100", "name": "n_iter"},
                                {"default": "10", "name": "n_seconds_sleep"},
                                {"default": "null", "name": "duration"},
                            ]
                        },
                        "metadata": {
                            "labels": {
                                "torch-job": "torch-ddp-0",
                                "torch-node": "1",
                            }
                        },
                        "name": "torch-ddp-1",
                        "outputs": {
                            "parameters": [
                                {
                                    "name": "duration",
                                    "value_from": {"path": "duration"},
                                }
                            ]
                        },
                        "pod_spec_patch": "topologySpreadConstraints:\n- maxSkew: 1\n  topologyKey: kubernetes.io/hostname\n  whenUnsatisfiable: DoNotSchedule\n  labelSelector:\n    matchExpressions:\n      - { key: torch-node, operator: In, values: ['0','1','2','3','4','5']}",
                        "retry_strategy": {
                            "limit": "1",
                            "retry_policy": "OnError",
                        },
                        "script": {
                            "command": ["python"],
                            "env": [
                                {"name": "NCCL_DEBUG", "value": "INFO"},
                                {
                                    "name": "bettmensch_ai_torch_ddp_min_nodes",
                                    "value": "2",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_max_nodes",
                                    "value": "2",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_node_rank",
                                    "value": "1",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_nproc_per_node",
                                    "value": "1",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_max_restarts",
                                    "value": "1",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_start_method",
                                    "value": "fork",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_rdzv_backend",
                                    "value": "static",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_rdzv_endpoint_url",
                                    "value": "torch-ddp-0-{{workflow.uid}}.argo.svc.cluster.local",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_rdzv_endpoint_port",
                                    "value": "29200",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_run_id",
                                    "value": "1",
                                },
                                {
                                    "name": "bettmensch_ai_torch_ddp_tee",
                                    "value": "0",
                                },
                            ],
                            "image": "bettmensch88/bettmensch.ai-pytorch:3.11-latest",
                            "image_pull_policy": "Always",
                            "name": "",
                            "resources": {
                                "limits": {
                                    "cpu": "100m",
                                    "memory": "700Mi",
                                    "nvidia.com/gpu": "1",
                                },
                                "requests": {
                                    "cpu": "100m",
                                    "memory": "700Mi",
                                    "nvidia.com/gpu": "1",
                                },
                            },
                            "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: n_iter = json.loads(r'''{{inputs.parameters.n_iter}}''')\nexcept: n_iter = r'''{{inputs.parameters.n_iter}}'''\ntry: n_seconds_sleep = json.loads(r'''{{inputs.parameters.n_seconds_sleep}}''')\nexcept: n_seconds_sleep = r'''{{inputs.parameters.n_seconds_sleep}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\nfrom bettmensch_ai.pipelines.io import OutputParameter\nduration = OutputParameter(\"duration\")\n\ndef tensor_reduce(n_iter: InputParameter=100, n_seconds_sleep: InputParameter=10, duration: OutputParameter=None) -> None:\n    \"\"\"When decorated with the torch_component decorator, implements a\n    bettmensch_ai.TorchComponent that runs a torch DDP across pods and nodes in\n    your K8s cluster.\"\"\"\n    import time\n    from datetime import datetime as dt\n    import GPUtil\n    import torch\n    import torch.distributed as dist\n    from bettmensch_ai.pipelines.component.torch_ddp import LaunchContext\n    has_gpu = torch.cuda.is_available()\n    ddp_context = LaunchContext()\n    print(f'GPU present: {has_gpu}')\n    if has_gpu:\n        dist.init_process_group(backend='nccl')\n    else:\n        dist.init_process_group(backend='gloo')\n    for i in range(1, n_iter + 1):\n        time.sleep(n_seconds_sleep)\n        GPUtil.showUtilization()\n        a = torch.tensor([ddp_context.rank])\n        print(f'{i}/{n_iter}: @{dt.now()}')\n        print(f'{i}/{n_iter}: Backend {dist.get_backend()}')\n        print(f'{i}/{n_iter}: Global world size: {ddp_context.world_size}')\n        print(f'{i}/{n_iter}: Global worker process rank: {ddp_context.rank}')\n        print(f'{i}/{n_iter}: This makes me worker process {ddp_context.rank + 1}/{ddp_context.world_size} globally!')\n        print(f'{i}/{n_iter}: Local rank of worker: {ddp_context.local_rank}')\n        print(f'{i}/{n_iter}: Local world size: {ddp_context.local_world_size}')\n        print(f'{i}/{n_iter}: This makes me worker process {ddp_context.local_rank + 1}/{ddp_context.local_world_size} locally!')\n        print(f'{i}/{n_iter}: Node/pod rank: {ddp_context.group_rank}')\n        if has_gpu:\n            device = torch.device(f'cuda:{ddp_context.local_rank}')\n            device_count = torch.cuda.device_count()\n            print(f'{i}/{n_iter}: GPU count: {device_count}')\n            device_name = torch.cuda.get_device_name(ddp_context.local_rank)\n            print(f'{i}/{n_iter}: GPU name: {device_name}')\n            device_property = torch.cuda.get_device_capability(device)\n            print(f'{i}/{n_iter}: GPU property: {device_property}')\n        else:\n            device = torch.device('cpu')\n        a_placed = a.to(device)\n        print(f'{i}/{n_iter}: Pre-`all_reduce` tensor: {a_placed}')\n        dist.all_reduce(a_placed)\n        print(f'{i}/{n_iter}: Post-`all_reduce` tensor: {a_placed}')\n        print('===================================================')\n    if duration is not None:\n        duration_seconds = n_iter * n_seconds_sleep\n        duration.assign(duration_seconds)\n\nfrom torch.distributed.elastic.multiprocessing.errors import record\n\ntensor_reduce=record(tensor_reduce)\n\nfrom bettmensch_ai.pipelines.component import as_torch_ddp\n\ntorch_ddp_decorator=as_torch_ddp()\n\ntorch_ddp_function=torch_ddp_decorator(tensor_reduce)\n\n\ntorch_ddp_function(n_iter,n_seconds_sleep,duration)",
                        },
                        "tolerations": [
                            {
                                "effect": "NoSchedule",
                                "key": "nvidia.com/gpu",
                                "operator": "Exists",
                            }
                        ],
                    },
                    {
                        "inputs": {"parameters": [{"name": "a"}]},
                        "metadata": {},
                        "name": "show-duration-param",
                        "outputs": {},
                        "retry_strategy": {
                            "limit": "1",
                            "retry_policy": "OnError",
                        },
                        "script": {
                            "command": ["python"],
                            "image": "bettmensch88/bettmensch.ai-standard:3.11-latest",
                            "image_pull_policy": "Always",
                            "name": "",
                            "resources": {
                                "limits": {"cpu": "100m", "memory": "100Mi"},
                                "requests": {"cpu": "100m", "memory": "100Mi"},
                            },
                            "source": "import os\nimport sys\nsys.path.append(os.getcwd())\n\n# --- preprocessing\nimport json\ntry: a = json.loads(r'''{{inputs.parameters.a}}''')\nexcept: a = r'''{{inputs.parameters.a}}'''\n\nfrom bettmensch_ai.pipelines.io import InputParameter\n\ndef show_parameter(a: InputParameter) -> None:\n    \"\"\"When decorated with the bettmensch_ai.components.component decorator,\n    implements a bettmensch_ai.Component that prints the values of its\n    InputParameter.\"\"\"\n    print(f'Content of input parameter a is: {a}')\n\nshow_parameter(a)\n",
                        },
                    },
                    {
                        "dag": {
                            "tasks": [
                                {
                                    "arguments": {
                                        "parameters": [
                                            {
                                                "name": "n_iter",
                                                "value": "{{workflow.parameters.n_iter}}",
                                            },
                                            {
                                                "name": "n_seconds_sleep",
                                                "value": "{{workflow.parameters.n_seconds_sleep}}",
                                            },
                                        ]
                                    },
                                    "name": "bettmensch-ai-inner-dag",
                                    "template": "bettmensch-ai-inner-dag",
                                }
                            ]
                        },
                        "inputs": {},
                        "metadata": {},
                        "name": "bettmensch-ai-outer-dag",
                        "outputs": {},
                    },
                ],
                "workflow_template_ref": {
                    "name": "pipeline-test-torch-gpu-pipeline-7c4zp"
                },
            },
            "task_results_completion_status": {
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-1368447231": True,
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-1861925387": True,
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-2020597252": True,
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-41628430": True,
                "pipeline-test-torch-gpu-pipeline-7c4zp-flow-9ldcf-947069694": True,
            },
        },
    )
