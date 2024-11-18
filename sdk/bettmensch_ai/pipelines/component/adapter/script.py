from bettmensch_ai.pipelines.constants import COMPONENT_IMPLEMENTATION
from hera.workflows import Script


class BettmenschAIAdapterOutScript(Script):
    implementation: str = COMPONENT_IMPLEMENTATION.adapter_out.value


class BettmenschAIAdapterInScript(Script):
    implementation: str = COMPONENT_IMPLEMENTATION.adapter_in.value
