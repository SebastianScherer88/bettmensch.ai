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
    base = f"{DOCKER_IMAGE_BASE}:3.11-latest"
    standard = f"{DOCKER_IMAGE_BASE}:3.11-latest"
    torch = f"{DOCKER_IMAGE_BASE}-pytorch:3.11-latest"
    lightning = f"{DOCKER_IMAGE_BASE}-pytorch-lightning:3.11-latest"
    annotated_transformer = (
        f"{DOCKER_IMAGE_BASE}-annotated-transformer:3.11-latest"
    )


INPUT_TYPE = "inputs"
OUTPUT_TYPE = "outputs"

PIPELINE_TYPE = "workflow"
COMPONENT_TYPE = "tasks"

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
    torch_ddp: str = "torch-ddp"


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
