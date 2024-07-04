import copy
import inspect
import textwrap
from typing import Callable, Dict, List, Tuple, Union
from uuid import uuid4

from bettmensch_ai.component import (
    ComponentInlineScriptRunner,
    _pipeline_context,
)
from bettmensch_ai.constants import PIPELINE_TYPE
from bettmensch_ai.io import (
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
)
from bettmensch_ai.torch_utils import (
    create_torch_service_template,
    delete_torch_service_template,
)
from bettmensch_ai.utils import (
    COMPONENT_BASE_IMAGE,
    BettmenschAITorchScript,
    bettmensch_ai_script,
    get_func_args,
    validate_func_args,
)
from hera.shared import global_config
from hera.workflows import Env, Label, Resource, Script, Task
from hera.workflows._context import _context
from hera.workflows._unparse import roundtrip
from hera.workflows.models import ImagePullPolicy


class TorchComponentInlineScriptRunner(ComponentInlineScriptRunner):

    """
    A customised version of the TorchComponentInlineScriptRunner that adds the
    decoration of the callable with the
    bettmensch_ai.torch_utils.torch_distribute decorator.
    """

    def generate_source(self, instance: Script) -> str:
        """Assembles and returns a script representation of the given function.

        This also assembles any extra script material prefixed to the string source.
        The script is expected to be a callable function the client is interested in submitting
        for execution on Argo and the `script_extra` material represents the parameter loading part obtained, likely,
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
        # in order to have consistent looking functions and getting rid of any comments
        # parsing issues.
        # See https://github.com/argoproj-labs/hera/issues/572
        content = roundtrip(
            textwrap.dedent(inspect.getsource(instance.source))
        ).splitlines()
        for i, line in enumerate(content):
            if line.startswith("def") or line.startswith("async def"):
                break

        # function definition + decoration with `torch_distribute`
        torch_distribute_decoration = [
            "from bettmensch_ai import torch_distribute\n",
            "@torch_distribute()",
        ]
        s = "\n".join(torch_distribute_decoration + content[i:])
        script += s

        # add function call
        script += f"\n\n{instance.source.__name__}"
        script += "(" + ",".join(args) + ")"

        return textwrap.dedent(script)


global_config.set_class_defaults(
    BettmenschAITorchScript, constructor=TorchComponentInlineScriptRunner()
)


class TorchComponent(object):

    type = "tasks"
    name: str = None
    func: Callable = None
    base_name: str = None
    name: str = None
    hera_template_kwargs: Dict = {}
    n_nodes: int
    min_nodes: int
    max_nodes: int
    nproc_per_node: int
    template_inputs: Dict[str, Union[InputParameter, InputArtifact]] = None
    template_outputs: Dict[str, Union[OutputParameter, OutputArtifact]] = None
    task_inputs: Dict[str, Union[InputParameter, InputArtifact]] = None
    service_templates: Dict[str, Callable] = None
    task_factory: List[Callable] = None
    k8s_namespace: str = "argo"
    k8s_service_name: str = ""

    def __init__(
        self,
        func: Callable,
        name: str = "",
        hera_template_kwargs: Dict = {},
        n_nodes: int = 1,
        min_nodes: int = 1,
        max_nodes: int = 1,
        nproc_per_node: int = 1,
        **component_inputs_kwargs: Union[
            InputParameter, OutputParameter, OutputArtifact
        ],
    ):

        self.hera_template_kwargs = hera_template_kwargs
        self.n_nodes = n_nodes
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.nproc_per_node = nproc_per_node
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
        self.k8s_service_name = f"{self.name}-{uuid4()}"
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

    def build_service_templates(self) -> Dict[str, Callable]:

        return {
            "create": create_torch_service_template(
                component_base_name=self.base_name,
                service_name=self.k8s_service_name,
            ),
            "delete": delete_torch_service_template(
                component_base_name=self.base_name,
                service_name=self.k8s_service_name,
            ),
        }

    def build_hera_task_factory(self) -> List[Callable]:
        """Generates the task factory task_wrapper callable from the
        hera.workflows.script decorator definition. Needs to be called outide
        of an active hera context.

        Returns:
            List[Callable]: A list of callables which, if called inside an
                active hera DAG context, generate the hera Tasks, one for each
                node in the distributed torch run.
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

        # add torch run environment variables to script kwargs
        task_factory = []

        for torch_node_rank in range(self.n_nodes):
            script_decorator_kwargs["env"] = [
                Env(
                    name="bettmensch_ai_distributed_torch_min_nodes",
                    value=self.min_nodes,
                ),
                Env(
                    name="bettmensch_ai_distributed_torch_max_nodes",
                    value=self.max_nodes,
                ),
                Env(
                    name="bettmensch_ai_distributed_torch_node_rank",
                    value=torch_node_rank,
                ),
                Env(
                    name="bettmensch_ai_distributed_torch_nproc_per_node",
                    value=self.nproc_per_node,
                ),
                Env(
                    name="bettmensch_ai_distributed_torch_rdvz_endpoint_url",
                    value=f"{self.k8s_service_name}.{self.k8s_namespace}.svc.cluster.local",
                ),
            ]

            script_decorator_kwargs[
                "name"
            ] = f"{self.base_name}-{torch_node_rank}"

            script_decorator_kwargs["labels"] = {"torch-node": torch_node_rank}

            if torch_node_rank == 0:
                script_decorator_kwargs["labels"]["app"] = self.k8s_service_name

            # this will invoke our custom TorchComponentInlineScriptRunner under the hood
            script_wrapper = bettmensch_ai_script(
                torch_component=True, **script_decorator_kwargs
            )

            task_node_factory = script_wrapper(func=self.func)
            task_factory.append(task_node_factory)

        return task_factory

    def to_hera(self) -> Tuple[Resource, List[Task]]:
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

        # create the torch service creation template and task
        create_service = Task(
            name=self.service_templates["create"].name,
            template=self.service_templates["create"].name,
        )

        distributed_tasks = []

        for torch_node_rank in range(self.n_nodes):

            name_i = (
                self.name
                if torch_node_rank == 0
                else f"{self.name}-worker-{torch_node_rank}"
            )

            distributed_tasks.append(
                self.task_factory[torch_node_rank](
                    arguments=[
                        task_input.to_hera()
                        for task_input in self.task_inputs.values()
                    ],
                    name=name_i,
                    depends=f"{self.depends} && {create_service.name}"
                    if self.depends
                    else create_service.name,
                )
            )

        # delete the torch service creation template and task
        delete_service = Task(
            name=self.service_templates["delete"].name,
            template=self.service_templates["delete"].name,
            depends=distributed_tasks[0].name,
        )

        return (
            [
                create_service,
            ]
            + distributed_tasks
            + [
                delete_service,
            ]
        )


def torch_component(func: Callable) -> Callable:
    """Takes a calleable and generates a configured TorchComponent factory that
    will generate a TorchComponent version of the callable if invoked inside an
    active PipelineContext.

    Usage:

    ```python
    @bettmensch_ai.torch_component #-> component factory
    def add(a: InputParameter = 1, b: InputParameter = 2, sum: OutputParameter = None):
        sum.assign(a + b)
    ```

    Decorating the above `add` method returns a component factory that
    generates a TorchComponent class instance when called from within an active
    PipelineContext.
    """

    def torch_component_factory(
        name: str = "",
        hera_template_kwargs: Dict = {},
        n_nodes: int = 1,
        min_nodes: int = 1,
        max_nodes: int = 1,
        nproc_per_node: int = 1,
        **component_inputs_kwargs,
    ) -> TorchComponent:

        return TorchComponent(
            func=func,
            name=name,
            hera_template_kwargs=hera_template_kwargs,
            n_nodes=n_nodes,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            nproc_per_node=nproc_per_node,
            **component_inputs_kwargs,
        )

    return torch_component_factory
