from enum import Enum

from hera.workflows.models import RetryStrategy, Toleration
from pydantic_settings import BaseSettings, SettingsConfigDict


class ComponentImageSettings(BaseSettings):
    account: str = "bettmensch88"
    repo_base: str = "bettmensch.ai"

    model_config = SettingsConfigDict(env_prefix="bettmensch_ai_docker_")


component_image_settings = ComponentImageSettings()
DOCKER_IMAGE_BASE = (
    f"{component_image_settings.account}/{component_image_settings.repo_base}"
)


class COMPONENT_IMAGE(Enum):
    base = f"{DOCKER_IMAGE_BASE}-standard:3.11-latest"
    standard = f"{DOCKER_IMAGE_BASE}-standard:3.11-latest"
    adapter = f"{DOCKER_IMAGE_BASE}-adapter:3.11-latest"
    torch = f"{DOCKER_IMAGE_BASE}-pytorch:3.11-latest"
    lightning = f"{DOCKER_IMAGE_BASE}-pytorch-lightning:3.11-latest"
    annotated_transformer = (
        f"{DOCKER_IMAGE_BASE}-annotated-transformer:3.11-latest"
    )


S3_ARTIFACT_REPOSITORY_BUCKET = "bettmensch-ai-artifact-repository"
S3_ARTIFACT_REPOSITORY_PREFIX = "argo-workflows"
DDP_TASK_ALIAS = "torch-ddp-task"


class IOType(Enum):
    inputs: str = "inputs"
    outputs: str = "outputs"


class ArgumentType(Enum):
    parameter: str = "parameter"
    artifact: str = "artifact"


class ResourceType(Enum):
    pipeline: str = "workflow"
    component: str = "tasks"


GPU_FLAG = "nvidia.com/gpu"

DDP_PORT_NAME = "ddp"
DDP_PORT_NUMBER = 29200

ARGO_NAMESPACE = "argo"
POD_RETRY_STRATEGY = RetryStrategy(
    limit="1",
    retry_policy="OnError",  # this covers the karpenter node consolidation
    # based evictions of dag task node pods
)

GPU_TOLERATION = Toleration(
    effect="NoSchedule", key=GPU_FLAG, operator="Exists"
)


class COMPONENT_IMPLEMENTATION(Enum):
    base: str = "base"
    standard: str = "standard"
    adapter_out: str = "adapter-out"
    adapter_in: str = "adapter-in"
    wait_on_k8s_external: str = "wait-on-k8s-external"
    torch_ddp: str = "torch-ddp"


class PipelineDagTemplate(Enum):
    inner: str = "bettmensch-ai-inner-dag"
    outer: str = "bettmensch-ai-outer-dag"


class TORCH_DDP_SCRIPT_IMPLEMENTATION(Enum):
    pre_adapter_io: str = "pre_adapter_io"
    torch_ddp: str = "torch_ddp"
    post_adapter_io: str = "post_adapter_io"


class FLOW_LABEL(Enum):
    """A utility class for valid Flow label keys"""

    pipeline_name: str = "bettmensch.ai/pipeline-name"
    pipeline_id: str = "bettmensch.ai/pipeline-id"
    phase: str = "workflows.argoproj.io/phase"


class FLOW_PHASE(Enum):
    """A utility class for valid Flow phase label values"""

    pending: str = "Pending"
    running: str = "Running"
    succeeded: str = "Succeeded"
    failed: str = "Failed"
    error: str = "Error"
    unknown: str = ""
