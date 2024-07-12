import inspect
from typing import Callable, Dict, List, Union

from bettmensch_ai.constants import COMPONENT_TYPE, PIPELINE_TYPE
from bettmensch_ai.io import (
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
)
from bettmensch_ai.pipeline_context import _pipeline_context
from bettmensch_ai.utils import get_func_args, validate_func_args
from hera.workflows import Task


class BaseComponent(object):

    type = COMPONENT_TYPE
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
        for the template outputs. To be used in the `build_task_factory`
        method.

        Note that InputParameter type arguments dont need to be passed
        explicitly to hera's  @script decorator since they are inferred from
        the decorated function's argument spec automatically.

        Args:
            func (Callable): The function the we want to wrap in a Component.
            annotation_types:
                List[Union[InputArtifact,OutputParameter,OutputArtifact]]: The
                annotation types to extract.
        Returns:
            Dict[str,Union[
                    InputParameter,
                    InputArtifact,
                    OutputParameter,
                    OutputArtifact
                    ]
                ]: The component's template's inputs/outputs.
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
            inputs (Dict[str,Union[InputParameter, OutputParameter,
            OutputArtifact]]): The I/O instances that the Component's
                constructor receives.

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
            Dict[str,Union[InputParameter, InputArtifact]]: The component's
                inputs.
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
                # underlying function for this component only, we set the
                # pinned value
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

    def build_hera_task_factory(self) -> Union[Callable, List[Callable]]:
        """Generates the task factory task_wrapper callable from the
        hera.workflows.script decorator definition. Needs to be called outide
        of an active hera context.

        Returns:
            Union[Task,List[Callable]]: A callable or a list of callables
                which, if called inside an  active hera DAG context, generate
                the (list of) hera Task(s).
        """

        raise NotImplementedError(
            "The BaseComponent does not implement a "
            "`build_hera_task_factory` method."
        )

    def to_hera(self) -> Union[Task, List[Task]]:
        """Generates a hera.workflow.Task instance. Needs to be called from
            within an active hera context, specifically:
            - an outer layer hera.WorkflowTemplate context
            - an inner layer hera.DAG context.
        Otherwise the `task_factory` invocation won't return the
        hera.workflows.Task instance, and it wont be added to either
        hera.WorkflowTemplate or the hera.DAG.

        Returns:
            Union[Task,List[Task]]: A task or a list of tasks that implement(s)
                this BaseComponent subclass instance in the hera library.
        """

        raise NotImplementedError(
            "The BaseComponent does not implement a " "`to_hera` method."
        )
