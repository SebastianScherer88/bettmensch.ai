import copy
import textwrap

from bettmensch_ai.pipelines.component.base.inline_script_runner import (
    BaseComponentInlineScriptRunner,
)
from hera.shared import global_config

from .script import BettmenschAIAdapterInScript, BettmenschAIAdapterOutScript


class AdapterOutInlineScriptRunner(BaseComponentInlineScriptRunner):

    """
    A custom InlineScriptRunner class to support the
    BettmenschAIAdapterOutScript by generating the code snippet for input
    gathering and uploading to S3.
    """

    def _get_s3_upload_code(
        self, instance: BettmenschAIAdapterOutScript
    ) -> str:
        """
        Adapted from the `_get_param_script_portion`
        method of the `InlineScriptRunner` class. Generates the code for
        exporting the inputs to S3.

        Args:
            instance (BettmenschAIAdapterOutScript): The script instance

        Returns:
            s3_export (str): The S3 export code.
        """

        # inputs
        inputs = instance._build_inputs()

        if inputs is None:
            raise ValueError(
                "This component type requires at least 1 input of"
                "type InputParameter or InputArtifact."
            )
        else:
            s3_export = "\n# --- exporting to S3\n"
            s3_export += "\nfrom bettmensch_ai.pipelines.component.adapter import AdapterIO\n"  # noqa: E501
            s3_export += "\nadapter_io = AdapterIO()\n"

        # utility to upload all input parameters at once to S3
        input_parameters = inputs.parameters if inputs.parameters else []
        if input_parameters:
            s3_export += "adapter_io.hera_input_parameters_to_s3()\n"

        # loop over input artifacts to upload individually to S3
        input_artifacts = inputs.artifacts if inputs.artifacts else []
        if input_artifacts:
            for input_artifact in sorted(
                input_artifacts, key=lambda ia: ia.name
            ):
                s3_export += f"""{
                    input_artifact.name
                } = adapter_io.upload_artifact_to_s3("{
                    input_artifact.name
                }")\n"""

        # --- outputs
        # export hardcoded s3_prefix output parameter
        s3_export += (
            "\nfrom bettmensch_ai.pipelines.io import OutputParameter\n"
        )
        s3_export += "\ns3_prefix=OutputParameter('s3_prefix')"
        s3_export += "\ns3_prefix.assign(adapter_io.s3_prefix)"

        s3_export = textwrap.dedent(s3_export)

        return s3_export

    def generate_source(self, instance: BettmenschAIAdapterOutScript) -> str:
        """The core method of any InlineScriptRunner (sub)class. Calls on the
        `_get_s3_upload_code` method to generate and return S3 upload code

        Args:
            instance (BettmenschAIAdapterOutScript): The script instance.

        Returns:
            str: The finished inline script.
        """

        """
        Adapted from the `_get_param_script_portion`
        method of the `InlineScriptRunner` class. Generates the code for
        importing the outputs from S3.

        Args:
            instance (BettmenschAIAdapterOutScript): The script instance

        Returns:
            s3_export (str): The S3 export code.
        """
        if not callable(instance.source):
            assert isinstance(instance.source, str)
            return instance.source
        script = ""
        # Argo will save the script as a file and run it with cmd:
        # - python /argo/staging/script
        # However, this prevents the script from importing modules in its cwd,
        # since it's looking for files relative to the script path.
        # We fix this by appending the cwd path to sys:
        if instance.add_cwd_to_sys_path or self.add_cwd_to_sys_path:
            script = "import os\nimport sys\nsys.path.append(os.getcwd())\n"

        script_extra = self._get_s3_upload_code(instance)
        script += copy.deepcopy(script_extra)
        script += "\n"

        return textwrap.dedent(script)


class AdapterInInlineScriptRunner(BaseComponentInlineScriptRunner):
    """
    A custom InlineScriptRunner class to support the
    BettmenschAIAdapterInScript by generating the code snippet for output
    gathering and downloading from S3.
    """

    def _get_s3_download_code(
        self, instance: BettmenschAIAdapterInScript
    ) -> str:
        """
        Adapted from the `_get_param_script_portion`
        method of the `InlineScriptRunner` class. Generates the code for
        importing the outputs from S3.

        Args:
            instance (BettmenschAIAdapterInScript): The script instance

        Returns:
            s3_import (str): The S3 import code.
        """

        # --- inputs
        # import hardcoded s3_prefix input parameter
        s3_import = "\n# --- importing from S3\n"
        s3_import += "\nimport json\n"
        s3_import += """try: s3_prefix = json.loads(r'''{{{{inputs.parameters.s3_prefix}}}}''')\n"""  # noqa: E501
        s3_import += """except: s3_prefix = r'''{{{{inputs.parameters.s3_prefix}}}}'''\n"""  # noqa: E501

        # --- outputs
        outputs = instance._build_outputs()

        if outputs is None:
            raise ValueError(
                "This component type requires at least 1 output of"
                "type OutputParameter or OutputArtifact."
            )
        else:
            s3_import += "\nfrom bettmensch_ai.pipelines.component.adapter import AdapterIO\n"  # noqa: E501
            s3_import += "\nadapter_io = AdapterIO(s3_prefix=s3_prefix)\n"

        # utility to download all input parameters at once from S3 and make
        # them available on disk for the hera/argo context
        output_parameters = outputs.parameters if outputs.parameters else []
        if output_parameters:
            s3_import += "adapter_io.s3_to_hera_output_parameters()\n"

        # loop over input artifacts to upload individually to S3
        output_artifacts = outputs.artifacts if outputs.artifacts else []
        if output_artifacts:
            for input_artifact in sorted(
                output_artifacts, key=lambda ia: ia.name
            ):
                s3_import += f"""{
                    input_artifact.name
                } = adapter_io.download_artifact_from_s3("{
                    input_artifact.name
                }")\n"""

        s3_import = textwrap.dedent(s3_import)

        return s3_import

    def generate_source(self, instance: BettmenschAIAdapterOutScript) -> str:
        """The core method of any InlineScriptRunner (sub)class. Calls on the
        `_get_s3_upload_code` method to generate and return S3 upload code

        Args:
            instance (BettmenschAIAdapterOutScript): The script instance.

        Returns:
            str: The finished inline script.
        """
        if not callable(instance.source):
            assert isinstance(instance.source, str)
            return instance.source
        script = ""
        # Argo will save the script as a file and run it with cmd:
        # - python /argo/staging/script
        # However, this prevents the script from importing modules in its cwd,
        # since it's looking for files relative to the script path.
        # We fix this by appending the cwd path to sys:
        if instance.add_cwd_to_sys_path or self.add_cwd_to_sys_path:
            script = "import os\nimport sys\nsys.path.append(os.getcwd())\n"

        script_extra = self._get_s3_upload_code(instance)
        script += copy.deepcopy(script_extra)
        script += "\n"

        return textwrap.dedent(script)


global_config.set_class_defaults(
    BettmenschAIAdapterInScript,
    constructor=AdapterInInlineScriptRunner(),
)

global_config.set_class_defaults(
    BettmenschAIAdapterOutScript,
    constructor=AdapterOutInlineScriptRunner(),
)
