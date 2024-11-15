import copy
import inspect
import textwrap

from bettmensch_ai.pipelines.component.base.inline_script_runner import (
    BaseComponentInlineScriptRunner,
)
from bettmensch_ai.pipelines.component.torch_ddp.script import (
    BettmenschAITorchDDPPostAdapterScript,
    BettmenschAITorchDDPPreAdapterScript,
    BettmenschAITorchDDPScript,
)
from hera.shared import global_config
from hera.workflows import Script
from hera.workflows._unparse import roundtrip


class TorchDDPComponentInlineScriptRunner(BaseComponentInlineScriptRunner):

    """
    A custom InlineScriptRunner class to support all 3 custom script classes
    - BettmenschAITorchDDPPreAdapterScript
    - BettmenschAITorchDDPScript
    - BettmenschAITorchDDPPostAdapterScript
    by generating the code snippets for their respective tasks:
    - input gathering and uploading to S3 code,
    - downlaoding inputs from S3, wrapping the user defined function in a
        torchrun context and calling it, then uploading the outputs to S3
    - S3 download and output gathering code

    Checks the `implementation` attribute of the passed Script subclass
    instance to decide which code should be generated.
    """

    def generate_source(self, instance: Script) -> str:
        """Assembles and returns a script representation of the given function.

        This also assembles any extra script material prefixed to the string
        source. The script is expected to be a callable function the client is
        interested in submitting for execution on Argo and the `script_extra`
        material represents the parameter loading part obtained, likely,
        through `get_param_script_portion`.

        Returns:
        -------
        str
            Final formatted script.
        """
        if not callable(instance.source):
            assert isinstance(instance.source, str)
            return instance.source
        args = inspect.getfullargspec(instance.source).args
        script = ""
        # Argo will save the script as a file and run it with cmd:
        # - python /argo/staging/script
        # However, this prevents the script from importing modules in its cwd,
        # since it's looking for files relative to the script path.
        # We fix this by appending the cwd path to sys:
        if instance.add_cwd_to_sys_path or self.add_cwd_to_sys_path:
            script = "import os\nimport sys\nsys.path.append(os.getcwd())\n"

        script_extra = (
            self._get_param_script_portion(instance) if args else None
        )
        if script_extra:
            script += copy.deepcopy(script_extra)
            script += "\n"

        # We use ast parse/unparse to get the source code of the function
        # in order to have consistent looking functions and getting rid of any
        # comments parsing issues.
        # See https://github.com/argoproj-labs/hera/issues/572
        content = roundtrip(
            textwrap.dedent(inspect.getsource(instance.source))
        ).splitlines()
        for i, line in enumerate(content):
            if line.startswith("def") or line.startswith("async def"):
                break

        # add function definition and decoration with `torch_ddp`
        torch_ddp_decoration = [
            "\nfrom torch.distributed.elastic.multiprocessing.errors import record\n",  # noqa: E501
            f"{instance.source.__name__}=record({instance.source.__name__})\n"
            "\nfrom bettmensch_ai.components import torch_ddp\n",
            "torch_ddp_decorator=torch_ddp()\n",
            f"""torch_ddp_function=torch_ddp_decorator({
                instance.source.__name__
            })\n""",
        ]
        function_definition = content[i:]
        s = "\n".join(function_definition + torch_ddp_decoration)
        script += s

        # add function call
        torch_ddp_function_call = (
            "\ntorch_ddp_function(" + ",".join(args) + ")"
        )
        script += torch_ddp_function_call

        return textwrap.dedent(script)


global_config.set_class_defaults(
    BettmenschAITorchDDPPreAdapterScript,
    constructor=TorchDDPComponentInlineScriptRunner(),
)

global_config.set_class_defaults(
    BettmenschAITorchDDPScript,
    constructor=TorchDDPComponentInlineScriptRunner(),
)

global_config.set_class_defaults(
    BettmenschAITorchDDPPostAdapterScript,
    constructor=TorchDDPComponentInlineScriptRunner(),
)
