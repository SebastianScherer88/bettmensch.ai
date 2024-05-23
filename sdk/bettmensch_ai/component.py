import copy
import inspect
import textwrap
from typing import Callable, Dict, List, Optional, Union

from bettmensch_ai.arguments import (
    ComponentInput,
    ComponentOutput,
    PipelineInput,
)
from bettmensch_ai.utils import (
    COMPONENT_BASE_IMAGE,
    get_func_args,
    validate_func_args,
)
from hera.shared import global_config
from hera.workflows import (
    DAG,
    InlineScriptConstructor,
    Parameter,
    Script,
    Task,
    WorkflowTemplate,
    models,
    script,
)
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
        ensure uniqueness of identifier in the PipelineContext."""

        counter = len(
            [
                component
                for component in self.components
                if component_base_name in component.name
            ]
        )

        return counter

    def add_component(self, component):
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
        if self.active:
            self.components = []
        else:
            raise Exception(
                f"Unable to clear components from context - pipeline context "
                "is not active."
            )


_pipeline_context = PipelineContext()


class ComponentInlineScriptRunner(InlineScriptConstructor):

    """A custom script constructor that submits a script as a `source` to Argo.

    This script constructor has been adapted from hera's
    InlineScriptConstructor to
    - consider template input type annotations to disambiguate inputs and
        outputs,
    - add output instance initialization before the main function body
    - add output file writing after the main function body
    """

    add_cwd_to_sys_path: Optional[bool] = None

    def _get_script_preprocessing(self, instance: Script) -> str:
        """Constructs and returns a script that loads the parameters of the
        specified arguments. Adapted to avoid listing ComponentOutput annotated
        source arguments in inputs, and instead add a file write section to the
        correct local path.

        Since Argo passes parameters through `{{input.parameters.name}}` it can
        be very cumbersome for users to manage that. This creates a script that
        automatically imports json and loads/adds code to interpret each
        independent argument into the script.

        Returns:
        -------
        str
            The string representation of the script to load.
        """

        inputs = instance._build_inputs()
        outputs = instance._build_outputs()

        output_names = [out_param.name for out_param in outputs.parameters]
        actual_input_parameters = [
            input_param
            for input_param in inputs.parameters
            if input_param.name not in output_names
        ]
        # remove the ComponentOutput annotated inputs

        preprocess = "\n# --- preprocessing\nimport json\n"
        # input parameter import
        for param in sorted(
            actual_input_parameters or [], key=lambda x: x.name
        ):
            # Hera does not know what the content of the `InputFrom` is, coming
            # from another task. In some cases non-JSON encoded strings are
            # returned, which fail the loads, but they can be used as plain
            # strings which is why this captures that in an except. This is
            # only used for `InputFrom` cases as the extra payload of the
            # script is not necessary when regular input is set on the task via
            # `func_params`
            if param.value_from is None:
                preprocess += f"""try: {param.name} = json.loads(r'''{{{{inputs.parameters.{param.name}}}}}''')\n"""
                preprocess += f"""except: {param.name} = r'''{{{{inputs.parameters.{param.name}}}}}'''\n"""

        # output parameter initialization
        if outputs.parameters:
            preprocess += (
                "\nfrom bettmensch_ai.arguments import ComponentOutput\n"
            )
        for param in sorted(outputs.parameters or [], key=lambda x: x.name):
            preprocess += (
                f"""{param.name} = ComponentOutput("{param.name}")\n"""
            )

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
            self._get_script_preprocessing(instance) if args else None
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

        s = "\n".join(content[i + 1 :])
        script += textwrap.dedent(s)
        return textwrap.dedent(script)


global_config.set_class_defaults(
    Script, constructor=ComponentInlineScriptRunner()
)


class Component(object):

    type = "tasks"
    name: str = None
    original_func: Callable = None
    func: Callable = None
    base_name: str = None
    inputs: Dict[str, ComponentInput] = None
    outputs: Dict[str, ComponentOutput] = None
    task_factory: Callable = None

    def __init__(
        self,
        func: Callable,
        hera_template_kwargs: Dict = {},
        **component_inputs_kwargs: Union[PipelineInput, ComponentOutput],
    ):

        self.hera_template_kwargs = hera_template_kwargs
        self.build(func, component_inputs_kwargs)

    def build(
        self,
        func: Callable,
        component_inputs: Dict[str, Union[PipelineInput, ComponentOutput]],
    ):

        self.func = func
        self.base_name = func.__name__
        _pipeline_context.add_component(self)
        self.inputs = self.generate_inputs_from_func(func, component_inputs)
        self.outputs = self.generate_outputs_from_func(func)
        self.task_factory = self.build_hera_task_factory()

    @property
    def parameter_owner_name(self) -> str:
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
            input.source.owner.name
            for input in self.inputs.values()
            if (input.source is not None)
            and (input.source.owner.type != "workflow")
        ]
        depends_deduped = list(set(depends))

        return " && ".join(depends_deduped)

    def generate_name(self, n: int):
        """Utility method to invoke by the global PipelineContext to generate a
        context wide unique identifier for the task node."""

        return f"{self.base_name}-{n}"

    def generate_inputs_from_func(
        self,
        func: Callable,
        inputs: Dict[str, Union[PipelineInput, ComponentOutput]],
    ) -> Dict[str, ComponentInput]:
        """Generates component inputs from the underlying function as well as
        the Component's constructor method's calls kwargs. Also
        - checks for correct ComponentInput type annotations in the decorated
            original function
        - ensures all original function inputs without default values are being
            specified

        Args:
            func (Callable): The function the we want to wrap in a Component.
            inputs (Dict[str,Union[PipelineInput,ComponentOutput]]): The
                PipelineInput or ComponentOutput instances that the Component's
                constructor receives.

        Raises:
            TypeError: Raised if any of the inputs arent of the supported types
                [PipelineInput,ComponentOutput]
            ValueError: Raised if the component is given an input that cannot
                be mapped onto any the underlying function's arguments that
                have been annotated with the InputParameter type.
            Exception: Raised if the component is not given an input for at
                least one of the underlying function's arguments without
                default value.

        Returns:
            Dict[str,ComponentInput]: The component's inputs.
        """

        validate_func_args(
            func, argument_types=[ComponentInput, ComponentOutput]
        )
        func_inputs = get_func_args(func, "annotation", [ComponentInput])
        non_default_args = get_func_args(func, "default", [inspect._empty])
        required_func_inputs = dict(
            [(k, v) for k, v in func_inputs.items() if k in non_default_args]
        )

        result = {}

        for name, input in inputs.items():

            if name not in func_inputs:
                raise ValueError(
                    f"Attempting to declare unknown component input {name}. "
                    f"Known inputs: {func_inputs}."
                )

            # assemble component input
            if isinstance(input, PipelineInput):
                component_input = ComponentInput(name=name, value=input.value)
            elif isinstance(input, ComponentOutput):
                component_input = ComponentInput(name=name)
            else:
                raise TypeError(
                    f"Input {input} must be of type PipelineInput or "
                    "ComponentOutput."
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

    def generate_outputs_from_func(
        self, func: Callable
    ) -> Dict[str, ComponentOutput]:
        """Generates the Component's outputs based on the underlying function's
        arguments annotated with the OutputParameter type.

        Args:
            func (Callable): The function the we want to wrap in a Component.

        Returns:
            Dict[str,ComponentOutput]: The component's outputs.
        """

        func_outputs = get_func_args(func, "annotation", [ComponentOutput])

        result = {}

        for output_name in func_outputs:
            component_output = ComponentOutput(name=output_name)
            component_output.set_owner(self)

            result[output_name] = component_output

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
                "outputs": [
                    Parameter(
                        name=output.name,
                        value_from=models.ValueFrom(path=output.path),
                    )
                    for output in self.outputs.values()
                ],
                "image_pull_policy": ImagePullPolicy.always,
            }
        )

        script_wrapper = script(**script_decorator_kwargs)

        task_factory = script_wrapper(func=self.func)

        return task_factory

    def to_hera_task(self) -> Task:
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
                input.to_hera_parameter() for input in self.inputs.values()
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
    @bettmensch_ai.component #-> component factory
    def add(a: ComponentInput = 1, b: ComponentInput = 2, sum: ComponentOutput = None):
        sum.assign(a + b)

    Decorating the above `add` method should return a component factory that
    - generates a Component class instance when called from within an active
        PipelineContext
      1. add a post function output processing step that ensures hera-compatible
        writing of output parameters to sensible local file paths
      2. add the inputs parameter type inputs 'a' and 'b' to the component
      3. add the parameter type output 'sum' to the component
          3.1 this should facilitate the reference the file path from step 1.
            in the `from` argument further downstream at the stage of mapping
            to a ArgoWorkflowTemplate
    """

    def component_factory(
        hera_template_kwargs: Dict = {}, **component_inputs_kwargs
    ) -> Component:

        return Component(
            func=func,
            hera_template_kwargs=hera_template_kwargs,
            **component_inputs_kwargs,
        )

    return component_factory


def test_hera_component():

    print("Testing component decoration")

    @component
    def add(
        a: ComponentInput = 1,
        b: ComponentInput = 2,
        sum: ComponentOutput = None,
    ) -> None:

        sum.assign(a + b)

    print(f"Created component factory: {add}")

    class MockPipeline:
        parameter_owner_name: str = "workflow"

    # mock active pipeline with 3 inputs
    pipeline_input_a = PipelineInput(name="a", value=1)
    pipeline_input_a.set_owner(MockPipeline())
    pipeline_input_b = PipelineInput(name="b", value=2)
    pipeline_input_b.set_owner(MockPipeline())
    pipeline_input_c = PipelineInput(name="c", value=3)
    pipeline_input_c.set_owner(MockPipeline())

    _pipeline_context.activate()
    _pipeline_context.clear()

    # add components to pipeline context
    a_plus_b = add(
        hera_template_kwargs={
            "image": "bettmensch88/bettmensch.ai:3.11-3d253c7"
        },
        a=pipeline_input_a,
        b=pipeline_input_b,
    )
    print(f"Created component: {a_plus_b.name}")

    a_plus_b_plus_c = add(
        hera_template_kwargs={
            "image": "bettmensch88/bettmensch.ai:3.11-3d253c7",
        },
        a=a_plus_b.outputs["sum"],
        b=pipeline_input_c,
    )
    print(f"Created component: {a_plus_b_plus_c.name}")

    # close pipeline context
    _pipeline_context.deactivate()

    with WorkflowTemplate(
        name="test_component",
        entrypoint="test_dag",
        namespace="argo",
        arguments=[
            Parameter(name="a"),
            Parameter(name="b"),
            Parameter(name="c"),
        ],
    ) as wft:

        with DAG(name="test_dag"):
            a_plus_b_task = a_plus_b.to_hera_task()
            a_plus_b_plus_c_task = a_plus_b_plus_c.to_hera_task()

    wft.to_file(".")


if __name__ == "__main__":
    test_hera_component()
