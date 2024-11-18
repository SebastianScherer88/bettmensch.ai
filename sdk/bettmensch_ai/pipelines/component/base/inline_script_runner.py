import copy
import inspect
import textwrap
from typing import Optional, Type

from hera.workflows import InlineScriptConstructor, Script
from hera.workflows._unparse import roundtrip

from .script import BettmenschAIBaseScript


class BaseComponentInlineScriptRunner(InlineScriptConstructor):

    """
    A customised version of the InlineScriptConstructor that implements a
    modified `_get_param_script_portion` method to ensure proper handling of
    the SDK's I/O objects at runtime.
    """

    add_cwd_to_sys_path: Optional[bool] = None

    def _get_param_script_portion(
        self, instance: Type[BettmenschAIBaseScript]
    ) -> str:
        """
        Adapted from the `_get_param_script_portion`
        method of the `InlineScriptRunner` class. Generates the code
        implementing the I/O import and preprocessing for the Component's
        underlying function:

        If the underlying function has at least one argument annotated with
        `InputParameter`, the values will be obtained from reading respective
        json string representations and stored in local variables named after
        the input.

        If the underlying function has at least one argument annotated with
        `InputArtifact`, the class will be imported and an instance will be
        initialized for each argument. This will make the hera input
        `Artifact`'s  content accessible throught the `InputArtifact`
        instance's `path` property, allowing the user function to access the
        value from inside the original function's scope at runtime.

        If the underlying function has at least one argument annotated with
        `OutputParameter`, the class will be imported and an instance will be
        initialized for each argument. This will make the hera output
        `Parameter`'s content source file location available through the
        `OutputParameter` instance's `assign()` method, allowing the user
        function to write to this location from inside the original function's
        scope at runtime.

        If the underlying function has at least one argument annotated with
        `OutputArtifact`, the class will be imported and an instance will be
        initialized for each argument. This will make the hera output
        `Artifact`'s content source file location available through the
        `OutputArtifact` instance's `path` property, allowing the user function
        to write to this location from inside the original function's scope at
        runtime.

        Args:
            instance (Script): The Script instance holding

        Returns:
            str: The preprocessing code section that needs to be prepended to
                the component's underlying function's code.
        """

        # populate input related vars
        inputs = instance._build_inputs()
        if inputs is None:
            input_parameters = input_artifacts = []
        else:
            input_parameters = inputs.parameters if inputs.parameters else []
            input_artifacts = inputs.artifacts if inputs.artifacts else []

        # populate output related vars
        outputs = instance._build_outputs()
        if outputs is None:
            output_parameters = output_artifacts = output_names = []
        else:
            if outputs.parameters:
                output_parameters = outputs.parameters
            else:
                output_parameters = []

            output_artifacts = outputs.artifacts if outputs.artifacts else []
            output_names = [
                output_arg.name
                for output_arg in output_parameters + output_artifacts
            ]

        # remove the ComponentOutput annotated inputs

        preprocess = "\n# --- preprocessing\nimport json\n"
        # input parameter import
        for input_parameter in sorted(input_parameters, key=lambda x: x.name):
            if input_parameter.name in output_names:
                continue
            preprocess += f"""try: {
                input_parameter.name
            } = json.loads(r'''{{{{inputs.parameters.{
                    input_parameter.name
            }}}}}''')\n"""
            preprocess += f"""except: {
                input_parameter.name
            } = r'''{{{{inputs.parameters.{
                input_parameter.name
            }}}}}'''\n"""

        # technically only needed for the torch_component, but add here to
        # simplify subclassing for now
        if input_parameters:
            preprocess += "\nfrom bettmensch_ai.io import InputParameter\n"

        # input artifact initialization to provide user access to input
        # artifact file location
        if input_artifacts:
            preprocess += "\nfrom bettmensch_ai.io import InputArtifact\n"
            for input_artifact in sorted(
                input_artifacts, key=lambda ia: ia.name
            ):
                preprocess += f"""{
                    input_artifact.name
                } = InputArtifact("{
                    input_artifact.name
                }")\n"""

        # output parameter initialization
        if output_parameters:
            preprocess += "\nfrom bettmensch_ai.io import OutputParameter\n"
            for output_param in sorted(
                output_parameters, key=lambda op: op.name
            ):
                preprocess += f"""{
                    output_param.name
                } = OutputParameter("{
                    output_param.name
                }")\n"""

        # output artifact initialization to provide user access to output
        # artifact file location
        if output_artifacts:
            preprocess += "\nfrom bettmensch_ai.io import OutputArtifact\n"
            for output_artifact in sorted(
                output_artifacts, key=lambda oa: oa.name
            ):
                preprocess += f"""{
                    output_artifact.name
                } = OutputArtifact("{
                    output_artifact.name
                }")\n"""

        preprocess = (
            textwrap.dedent(preprocess)
            if preprocess != "\n# --- preprocessing\nimport json\n"
            else ""
        )

        return preprocess

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

        # add function definition
        function_definition = content[i:]
        script += "\n".join(function_definition)

        # add function call
        function_call = f"\n{instance.source.__name__}(" + ",".join(args) + ")"
        script += function_call

        return textwrap.dedent(script)
