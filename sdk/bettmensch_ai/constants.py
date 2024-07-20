from enum import Enum

from hera.workflows.models import RetryStrategy, Toleration


class COMPONENT_IMAGE(Enum):
    base = "bettmensch88/bettmensch.ai:3.11-latest"
    standard = "bettmensch88/bettmensch.ai:3.11-latest"
    torch = "bettmensch88/bettmensch.ai-torch:3.11-latest"
    lightning = "bettmensch88/bettmensch.ai-lightning:3.11-latest"


INPUT_TYPE = "inputs"
OUTPUT_TYPE = "outputs"

PIPELINE_TYPE = "workflow"
COMPONENT_TYPE = "tasks"

GPU_FLAG = "nvidia.com/gpu"

DDP_PORT_NAME = "ddp"
DDP_PORT_NUMBER = 29200

POD_RETRY_STRATEGY = RetryStrategy(
    limit=1,
    retry_policy="OnError",  # this covers the karpenter node consolidation
    # based evictions of dag task node pods
)

GPU_TOLERATION = Toleration(
    effect="NoSchedule", key=GPU_FLAG, operator="Exists"
)


class COMPONENT_IMPLEMENTATION(Enum):
    base: str = "base"
    standard: str = "standard"
    torch: str = "torch"
    lightning = "bettmensch88/bettmensch.ai-lightning:3.11"
