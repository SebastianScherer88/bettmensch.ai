import copy
import inspect
import textwrap
from typing import Callable, Dict, List, Optional, Union
from uuid import uuid4

from bettmensch_ai.components.base_component import BaseComponent
from bettmensch_ai.components.base_inline_script_runner import (
    BaseComponentInlineScriptRunner,
)
from bettmensch_ai.components.torch_utils import (
    create_torch_service_template,
    delete_torch_service_template,
)
from bettmensch_ai.constants import (
    COMPONENT_BASE_IMAGE,
    COMPONENT_IMPLEMENTATION,
    DDP_PORT_NAME,
    DDP_PORT_NUMBER,
)
from bettmensch_ai.io import InputParameter, OutputArtifact, OutputParameter
from bettmensch_ai.utils import BettmenschAITorchScript, bettmensch_ai_script
from hera.shared import global_config
from hera.workflows import Env, Script, Task
from hera.workflows._unparse import roundtrip
from hera.workflows.models import ContainerPort, ImagePullPolicy, Protocol


class TorchComponentInlineScriptRunner(BaseComponentInlineScriptRunner):

    """
    A customised version of the TorchComponentInlineScriptRunner that adds the
    decoration of the callable with the
    bettmensch_ai.torch_utils.torch_distribute decorator.
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

        # add function definition and decoration with `torch_distribute`
        torch_distribute_decoration = [
            "\nfrom bettmensch_ai.components import torch_distribute\n",
            "torch_distribute_decorator=torch_distribute()\n"
            f"""torch_distributed_function=torch_distribute_decorator({
                instance.source.__name__
            })\n""",
        ]
        function_definition = content[i:]
        s = "\n".join(function_definition + torch_distribute_decoration)
        script += s

        # add function call
        torch_distributed_function_call = (
            "\ntorch_distributed_function(" + ",".join(args) + ")"
        )
        script += torch_distributed_function_call

        return textwrap.dedent(script)


global_config.set_class_defaults(
    BettmenschAITorchScript, constructor=TorchComponentInlineScriptRunner()
)


class TorchComponent(BaseComponent):

    implementation: str = COMPONENT_IMPLEMENTATION.torch.value
    n_nodes: int
    min_nodes: int
    nproc_per_node: int
    service_templates: Dict[str, Callable] = None
    k8s_namespace: str = "argo"
    k8s_service_name: str = ""

    # if no resources are specified, set minimal requirements derived from
    # testing the ddp example on K8s
    cpu: Optional[Union[float, int, str]] = "100m"
    memory: Optional[str] = "300Mi"

    def __init__(
        self,
        func: Callable,
        name: str = "",
        hera_template_kwargs: Dict = {},
        n_nodes: int = 1,
        min_nodes: Optional[int] = None,
        nproc_per_node: int = 1,
        **component_inputs_kwargs: Union[
            InputParameter, OutputParameter, OutputArtifact
        ],
    ):

        self.set_node_specs(n_nodes, min_nodes)
        self.nproc_per_node = nproc_per_node
        super().__init__(
            func, name, hera_template_kwargs, **component_inputs_kwargs
        )

    def set_node_specs(self, n_nodes: int, min_nodes: Optional[int]):

        if min_nodes is not None:
            assert (
                n_nodes >= min_nodes
            ), f"""Mininum number of nodes {
                min_nodes
            } can not be greater than total number of nodes {n_nodes}."""
        else:
            min_nodes = n_nodes

        self.n_nodes = n_nodes
        self.min_nodes = min_nodes

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

        self.k8s_service_name = f"{self.name}-{uuid4()}"

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

        if "resources" not in script_decorator_kwargs:
            script_decorator_kwargs["resources"] = self.build_resources()

        if "tolerations" not in script_decorator_kwargs:
            script_decorator_kwargs["tolerations"] = self.build_tolerations()

        script_decorator_kwargs["ports"] = [
            ContainerPort(
                container_port=DDP_PORT_NUMBER,
                protocol=Protocol.tcp,
                name=DDP_PORT_NAME,
            )
        ]

        # add torch run environment variables to script kwargs
        task_factory = []

        for torch_node_rank in range(self.n_nodes):
            script_decorator_kwargs["env"] = [
                Env(
                    name="NCCL_DEBUG",
                    value="INFO",
                ),
                Env(
                    name="bettmensch_ai_distributed_torch_min_nodes",
                    value=self.min_nodes,
                ),
                Env(
                    name="bettmensch_ai_distributed_torch_max_nodes",
                    value=self.n_nodes,
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
                    name="bettmensch_ai_distributed_torch_max_restarts",
                    value=1,
                ),
                # torch's LaunchConfig's default of 'spawn' doesnt seem to work
                # inside the argo emissary runtime context for some reason
                Env(
                    name="bettmensch_ai_distributed_torch_start_method",
                    value="fork",
                ),
                Env(
                    name="bettmensch_ai_distributed_torch_rdzv_backend",
                    value="static",
                ),
                Env(
                    name="bettmensch_ai_distributed_torch_rdzv_endpoint_url",
                    value=f"{self.k8s_service_name}.{self.k8s_namespace}"
                    + ".svc.cluster.local",
                ),
                Env(
                    name="bettmensch_ai_distributed_torch_rdzv_endpoint_port",
                    value=DDP_PORT_NUMBER,
                ),
                Env(
                    name="bettmensch_ai_distributed_torch_run_id",
                    value=1,
                ),
                Env(
                    name="bettmensch_ai_distributed_torch_tee",
                    value=0,
                ),
            ]

            script_decorator_kwargs[
                "name"
            ] = f"{self.base_name}-{torch_node_rank}"

            labels: Dict = script_decorator_kwargs.get("labels", {})
            labels.update(
                {
                    "torch-node": torch_node_rank,
                    "torch-job": self.k8s_service_name,
                }
            )
            script_decorator_kwargs["labels"] = labels

            # this will invoke our custom TorchComponentInlineScriptRunner
            # under the hood
            script_wrapper = bettmensch_ai_script(
                torch_component=True, **script_decorator_kwargs
            )

            task_node_factory = script_wrapper(func=self.func)
            task_factory.append(task_node_factory)

        return task_factory

    def to_hera(self) -> List[Task]:
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


def torch_component(func: Callable) -> Callable[..., TorchComponent]:
    """Takes a calleable and generates a configured TorchComponent factory that
    will generate a TorchComponent version of the callable if invoked inside an
    active PipelineContext.

    Usage:

    ```python
    @bettmensch_ai.torch_component #-> component factory
    def add(
        a: InputParameter = 1,
        b: InputParameter = 2,
        sum: OutputParameter = None
    ):
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
        min_nodes: int = None,
        nproc_per_node: int = 1,
        **component_inputs_kwargs,
    ) -> TorchComponent:

        return TorchComponent(
            func=func,
            name=name,
            hera_template_kwargs=hera_template_kwargs,
            n_nodes=n_nodes,
            min_nodes=min_nodes,
            nproc_per_node=nproc_per_node,
            **component_inputs_kwargs,
        )

    return torch_component_factory
