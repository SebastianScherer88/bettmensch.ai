from bettmensch_ai.pipelines.constants import TORCH_DDP_SCRIPT_IMPLEMENTATION
from hera.workflows import Script


class BettmenschAITorchDDPPreAdapterScript(Script):
    implementation: str = TORCH_DDP_SCRIPT_IMPLEMENTATION.pre_adapter_io.value


class BettmenschAITorchDDPScript(Script):
    implementation: str = TORCH_DDP_SCRIPT_IMPLEMENTATION.torch_ddp.value


class BettmenschAITorchDDPPostAdapterScript(Script):
    implementation: str = TORCH_DDP_SCRIPT_IMPLEMENTATION.post_adapter_io.value
