import inspect

from bettmensch_ai.pipelines.component.base.inline_script_runner import (
    BaseComponentInlineScriptRunner,
)
from hera.shared import global_config

from .script import BettmenschAITorchDDPScript


class TorchDDPComponentInlineScriptRunner(BaseComponentInlineScriptRunner):

    """
    A customised version of the TorchComponentInlineScriptRunner that adds the
    decoration of the callable with the
    bettmensch_ai.torch_utils.torch_ddp decorator.
    """

    def _get_invocation_script_portion(
        self, instance: BettmenschAITorchDDPScript
    ) -> str:

        # ddp_torch decoration script portion
        decoration = [
            "\nfrom torch.distributed.elastic.multiprocessing.errors import record\n",  # noqa: E501
            f"{instance.source.__name__}=record({instance.source.__name__})\n"
            "\nfrom bettmensch_ai.pipelines.component import as_torch_ddp\n",
            "torch_ddp_decorator=as_torch_ddp()\n",
            f"""torch_ddp_function=torch_ddp_decorator({
                instance.source.__name__
            })\n""",
        ]

        # invocation script portion
        args = inspect.getfullargspec(instance.source).args
        invocation = [
            "\ntorch_ddp_function(" + ",".join(args) + ")",
        ]

        invocation_script = "\n".join(decoration + invocation)

        return invocation_script


global_config.set_class_defaults(
    BettmenschAITorchDDPScript,
    constructor=TorchDDPComponentInlineScriptRunner(),
)
