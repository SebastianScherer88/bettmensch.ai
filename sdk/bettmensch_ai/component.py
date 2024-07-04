import copy
import inspect
import textwrap
from typing import Callable, Dict, List, Optional, Union

from bettmensch_ai.constants import PIPELINE_TYPE
from bettmensch_ai.io import (
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
)
from bettmensch_ai.utils import (
    COMPONENT_BASE_IMAGE,
    get_func_args,
    validate_func_args,
)
from hera.shared import global_config
from hera.workflows import InlineScriptConstructor, Script, Task, script
from hera.workflows._unparse import roundtrip
from hera.workflows.models import ImagePullPolicy


class PipelineContext(object):
    """Globally accessible pipeline meta data storage utility."""

    _active: bool = False
    components: List = []

    @property
    def active(self):
        return self._active

    def activate(self):
        self._active = True

    def deactivate(self):
        self._active = False

    def get_component_name_counter(self, component_base_name: str) -> int:
        """Utility to get the counter for the name of a potential component to
        ensure uniqueness of identifier in the PipelineContext.

        Args:
            component_base_name (str): The `_base_name` attribute value of a
                Component instance.

        Returns:
            int: The unique counter of that Component w.r.t. the global
                PipelineContext.
        """

        counter = len(
            [
                component
                for component in self.components
                if component_base_name in component.name
            ]
        )

        return counter

    def add_component(self, component: "Component"):
        """Adds the specified Component instance to the global PipelineContext.

        Args:
            component (Component): The Component instance that will be added.

        Raises:
            Exception: Raised if this method was not called within an active
                PipelineContext.
        """
        if self.active:
            component_counter = self.get_component_name_counter(
                component.base_name
            )
            component.name = component.generate_name(component_counter)
            self.components.append(component)
        else:
            raise Exception(
                f"Unable to add component {component.base_name} - pipeline "
                "context is not active."
            )

    def clear(self):
        """Removes all components from the active PipelineContext. Useful when
        defining a (new) Pipeline and you want to ensure a clean slate.

        Raises:
            Exception: Raised if this method was not called within an active
                PipelineContext.
        """
        if self.active:
            self.components = []
        else:
            raise Exception(
                f"Unable to clear components from context - pipeline context "
                "is not active."
            )

    def __enter__(self):
        self._active = True

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):

        pass


_pipeline_context = PipelineContext()


class ComponentInlineScriptRunner(InlineScriptConstructor):

    """
    A customised version of the InlineScriptConstructor that implements a
    modified `_get_param_script_portion` method to ensure proper handling of the
    SDK's I/O objects at runtime.
    """

    add_cwd_to_sys_path: Optional[bool] = None

    def _get_param_script_portion(self, instance: Script) -> str:
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
        `Artifact`'s  content accessible throught the `InputArtifact` instance's
        `path` property, allowing the user function to access the value from
        inside the original function's scope at runtime.

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
            output_parameters = outputs.parameters if outputs.parameters else []
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
            preprocess += f"""try: {input_parameter.name} = json.loads(r'''{{{{inputs.parameters.{input_parameter.name}}}}}''')\n"""
            preprocess += f"""except: {input_parameter.name} = r'''{{{{inputs.parameters.{input_parameter.name}}}}}'''\n"""

        # input artifact initialization to provide user access to input artifact
        # file location
        if input_artifacts:
            preprocess += (
                "\nfrom bettmensch_ai.arguments import InputArtifact\n"
            )
            for input_artifact in sorted(input_artifacts, key=lambda x: x.name):
                preprocess += f"""{input_artifact.name} = InputArtifact("{input_artifact.name}")\n"""

        # output parameter initialization
        if output_parameters:
            preprocess += (
                "\nfrom bettmensch_ai.arguments import OutputParameter\n"
            )
            for output_param in sorted(output_parameters, key=lambda x: x.name):
                preprocess += f"""{output_param.name} = OutputParameter("{output_param.name}")\n"""

        # output artifact initialization to provide user access to output
        # artifact file location
        if output_artifacts:
            preprocess += (
                "\nfrom bettmensch_ai.arguments import OutputArtifact\n"
            )
            for output_artifact in sorted(
                output_artifacts, key=lambda x: x.name
            ):
                preprocess += f"""{output_artifact.name} = OutputArtifact("{output_artifact.name}")\n"""

        preprocess = (
            textwrap.dedent(preprocess)
            if preprocess != "\n# --- preprocessing\nimport json\n"
            else ""
        )

        return preprocess


global_config.set_class_defaults(
    Script, constructor=ComponentInlineScriptRunner()
)


class Component(object):

    type = "tasks"
    name: str = None
    func: Callable = None
    base_name: str = None
    name: str = None
    hera_template_kwargs: Dict = {}
    template_inputs: Dict[str, Union[InputParameter, InputArtifact]] = None
    template_outputs: Dict[str, Union[OutputParameter, OutputArtifact]] = None
    task_inputs: Dict[str, Union[InputParameter, InputArtifact]] = None
    task_factory: Callable = None

    def __init__(
        self,
        func: Callable,
        name: str = "",
        hera_template_kwargs: Dict = {},
        **component_inputs_kwargs: Union[
            InputParameter, OutputParameter, OutputArtifact
        ],
    ):

        self.hera_template_kwargs = hera_template_kwargs
        self.build(func, component_inputs_kwargs, name=name)

    def build(
        self,
        func: Callable,
        task_inputs: Dict[
            str, Union[InputParameter, OutputParameter, OutputArtifact]
        ],
        name: str = "",
    ):

        self.func = func

        base_name = func.__name__ if not name else name
        self.base_name = base_name.replace("_", "-")
        _pipeline_context.add_component(self)
        validate_func_args(
            func,
            argument_types=[
                InputParameter,
                OutputParameter,
                InputArtifact,
                OutputArtifact,
            ],
        )
        self.template_inputs = self.build_template_ios(func, (InputArtifact,))
        self.template_outputs = self.build_template_ios(
            func, (OutputParameter, OutputArtifact)
        )
        self.task_inputs = self.build_task_inputs(func, task_inputs)
        self.task_factory = self.build_hera_task_factory()

    @property
    def io_owner_name(self) -> str:
        return f"{self.type}.{self.name}"

    @property
    def depends(self) -> str:
        """Generates hera compatible dependency string from the inputs of the
        Component instance.

        Returns:
            str: A hera compatible task dependency string for the `depends`
                field of a workflow template task
        """
        depends = [
            task_input.source.owner.name
            for task_input in self.task_inputs.values()
            if getattr(task_input.source.owner, "type", None)
            not in (PIPELINE_TYPE, None)
        ]
        depends_deduped = list(set(depends))

        return " && ".join(depends_deduped)

    @property
    def outputs(self) -> Dict[str, Union[OutputParameter, OutputArtifact]]:
        return self.template_outputs

    def generate_name(self, n: int):
        """Utility method to invoke by the global PipelineContext to generate a
        context wide unique identifier for the task node."""

        return f"{self.base_name}-{n}"

    def build_template_ios(
        self,
        func: Callable,
        annotation_types: List[
            Union[InputArtifact, OutputParameter, OutputArtifact]
        ],
    ) -> Dict[str, Union[InputArtifact, OutputParameter, OutputArtifact]]:
        """Builds the Component's template's inputs/outputs based on the
        underlying function's arguments annotated with the
        - InputParameter for the template inputs or
        - OutputsParameter or the
        - OutputArtifact
        for the template outputs. To be used in the `build_task_factory` method.

        Note that InputParameter type arguments dont need to be passed
        explicitly to hera's  @script decorator since they are inferred from the
        decorated function's argument spec automatically.

        Args:
            func (Callable): The function the we want to wrap in a Component.
            annotation_types: List[Union[InputArtifact,OutputParameter,OutputArtifact]]: The annotation types to extract.
        Returns:
            Dict[str,Union[InputParameter,InputArtifact,OutputParameter,OutputArtifact]]: The component's
                template's inputs/outputs.
        """

        func_ios = get_func_args(func, "annotation", annotation_types)

        template_ios = {}

        for io_name, io_param in func_ios.items():
            template_io = io_param.annotation(name=io_name)
            template_io.set_owner(self)

            template_ios[io_name] = template_io

        return template_ios

    def build_task_inputs(
        self,
        func: Callable,
        task_inputs: Dict[
            str, Union[InputParameter, OutputParameter, OutputArtifact]
        ],
    ) -> Dict[str, Union[InputParameter, InputArtifact]]:
        """Builds the Component's task's inputs from the spec passed during the
        DAG construction phase. Also ensures all InputParameter without default
        values are being specified

        Args:
            func (Callable): The function the we want to wrap in a Component.
            inputs (Dict[str,Union[InputParameter, OutputParameter, OutputArtifact]]): The
                I/O instances that the Component's constructor receives.

        Raises:
            TypeError: Raised if any of the inputs arent of the supported types
                [InputParameter, OutputParameter, OutputArtifact]
            ValueError: Raised if the component is given an input that cannot
                be mapped onto any the underlying function's arguments that
                have been annotated with the InputParameter type.
            Exception: Raised if the component is not given an input for at
                least one of the underlying function's arguments without
                default value.

        Returns:
            Dict[str,Union[InputParameter, InputArtifact]]: The component's inputs.
        """

        func_inputs = get_func_args(
            func, "annotation", [InputParameter, InputArtifact]
        )
        non_default_args = get_func_args(func, "default", [inspect._empty])
        required_func_inputs = dict(
            [(k, v) for k, v in func_inputs.items() if k in non_default_args]
        )

        result = {}

        for name, input in task_inputs.items():

            if name not in func_inputs:
                raise ValueError(
                    f"Attempting to declare unknown component input {name}. "
                    f"Known inputs: {func_inputs}."
                )

            # assemble component input
            if isinstance(input, InputParameter):
                # for pipeline inputs, we retain the (possible) default value.
                # for a hardcoded component input pinning the argument of the
                # underlying function for this component only, we set the pinned
                # value
                component_input = InputParameter(name=name, value=input.value)
            elif isinstance(input, OutputParameter):
                # a component output won't have a default value to retain. the
                # input's value will be the hera reference expression
                component_input = InputParameter(name=name)
            elif isinstance(input, OutputArtifact):
                component_input = InputArtifact(name=name)
            else:
                raise TypeError(
                    f"Input {input} must be of one of  "
                    "(InputParamter, OutputParameter, OutputArtifact)"
                )

            component_input.set_source(input)
            component_input.set_owner(self)

            # remove declared input from required inputs (if relevant)
            if name in required_func_inputs:
                del required_func_inputs[name]

            result[name] = component_input

        # ensure no required inputs are left unspecified
        if required_func_inputs:
            raise Exception(
                f"Unspecified required input(s) left: {required_func_inputs}"
            )

        return result

    def build_hera_task_factory(self) -> Callable:
        """Generates the task factory task_wrapper callable from the
        hera.workflows.script decorator definition. Needs to be called outide
        of an active hera context.

        Returns:
            Task: A task that implements this Component instance in the hera
                library.
        """

        script_decorator_kwargs = self.hera_template_kwargs.copy()
        script_decorator_kwargs.update(
            {
                "inputs": [
                    template_input.to_hera(template=True)
                    for template_input in self.template_inputs.values()
                ],
                "outputs": [
                    template_output.to_hera()
                    for template_output in self.template_outputs.values()
                ],
                "name": self.base_name,
            }
        )

        if "image" not in script_decorator_kwargs:
            script_decorator_kwargs["image"] = COMPONENT_BASE_IMAGE

        if "image_pull_policy" not in script_decorator_kwargs:
            script_decorator_kwargs[
                "image_pull_policy"
            ] = ImagePullPolicy.always

        # this will invoke our custom ComponentInlineScriptRunner under the hood
        script_wrapper = script(**script_decorator_kwargs)

        task_factory = script_wrapper(func=self.func)

        return task_factory

    def to_hera(self) -> Task:
        """Generates a hera.workflow.Task instance. Needs to be called from
            within an active hera context, specifically:
            - an outer layer hera.WorkflowTemplate context
            - an inner layer hera.DAG context.
        Otherwise the `task_factory` invocation won't return the
        hera.workflows.Task instance, and it wont be added to either
        hera.WorkflowTemplate or the hera.DAG.

        Returns:
            Task: A task that implements this Component instance in the hera
                library.
        """
        task = self.task_factory(
            arguments=[
                task_input.to_hera() for task_input in self.task_inputs.values()
            ],
            name=self.name,
            depends=self.depends,
        )

        return task


def component(func: Callable) -> Callable:
    """Takes a calleable and generates a configured Component factory that will
    generate a Component version of the callable if invoked inside an active
    PipelineContext.

    Usage:

    ```python
    @bettmensch_ai.component #-> component factory
    def add(a: InputParameter = 1, b: InputParameter = 2, sum: OutputParameter = None):
        sum.assign(a + b)
    ```

    Decorating the above `add` method returns a component factory that
    generates a Component class instance when called from within an active
    PipelineContext.
    """

    def component_factory(
        name: str = "",
        hera_template_kwargs: Dict = {},
        **component_inputs_kwargs,
    ) -> Component:

        return Component(
            func=func,
            name=name,
            hera_template_kwargs=hera_template_kwargs,
            **component_inputs_kwargs,
        )

    return component_factory
