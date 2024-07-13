from hera.workflows.models import Toleration

INPUT_TYPE = "inputs"
OUTPUT_TYPE = "outputs"

PIPELINE_TYPE = "workflow"
COMPONENT_TYPE = "tasks"

GPU_FLAG = "nvidia.com/gpu"

DDP_PORT_NAME = "ddp"
DDP_PORT_NUMBER = 29200

GPU_TOLERATION = Toleration(
    effect="NoSchedule", key=GPU_FLAG, operator="Exists"
)
