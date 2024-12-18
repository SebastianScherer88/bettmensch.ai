from typing import Callable, Dict, List, Optional, Type, Union

from bettmensch_ai.pipelines.component.base import BettmenschAIBaseScript
from bettmensch_ai.pipelines.component.base.component import BaseComponent
from bettmensch_ai.pipelines.component.torch_ddp.utils import (
    LaunchConfigSettings,
    create_torch_ddp_service_template,
    delete_torch_ddp_service_template,
)
from bettmensch_ai.pipelines.component.utils import bettmensch_ai_script
from bettmensch_ai.pipelines.constants import (
    ARGO_NAMESPACE,
    COMPONENT_IMAGE,
    COMPONENT_IMPLEMENTATION,
    DDP_PORT_NAME,
    DDP_PORT_NUMBER,
)
from bettmensch_ai.pipelines.io import (
    InputParameter,
    OutputArtifact,
    OutputParameter,
)
from hera.workflows import Env, Task
from hera.workflows.models import ContainerPort, Protocol

from .script import BettmenschAITorchDDPScript


class TorchDDPComponent(BaseComponent):

    implementation: str = COMPONENT_IMPLEMENTATION.torch_ddp.value
    default_image: str = COMPONENT_IMAGE.torch.value
    script: Type[BettmenschAIBaseScript] = BettmenschAITorchDDPScript
    n_nodes: int
    min_nodes: int
    nproc_per_node: int
    service_templates: Dict[str, Callable] = None
    k8s_namespace: str = ARGO_NAMESPACE

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
            "create": create_torch_ddp_service_template(
                component_base_name=self.base_name,
                component_task_name=self.name,
            ),
            "delete": delete_torch_ddp_service_template(
                component_base_name=self.base_name,
                component_task_name=self.name,
            ),
        }

    def build_script_decorator_kwargs(self, torch_node_rank: int) -> Dict:
        """Extending the base method to consider the parallisation of the node
        across multiple tasks = pods.

        Returns:
            Dict: The keyword arguments to be unravelled into the
                bettmensch_ai.utils.bettmensch_ai_script decorator when
                building components.
        """

        script_decorator_kwargs = super().build_script_decorator_kwargs()

        if torch_node_rank == 0:
            script_decorator_kwargs["ports"] = [
                ContainerPort(
                    container_port=DDP_PORT_NUMBER,
                    protocol=Protocol.tcp,
                    name=DDP_PORT_NAME,
                )
            ]

        # add torch run environment variables to script kwargs
        script_decorator_kwargs["env"] = [
            Env(
                name="NCCL_DEBUG",
                value="INFO",
            ),
            Env(
                name=f"{LaunchConfigSettings.model_config['env_prefix']}min_nodes",  # noqa: E501
                value=self.min_nodes,
            ),
            Env(
                name=f"{LaunchConfigSettings.model_config['env_prefix']}max_nodes",  # noqa: E501
                value=self.n_nodes,
            ),
            Env(
                name=f"{LaunchConfigSettings.model_config['env_prefix']}node_rank",  # noqa: E501
                value=torch_node_rank,
            ),
            Env(
                name=f"{LaunchConfigSettings.model_config['env_prefix']}nproc_per_node",  # noqa: E501
                value=self.nproc_per_node,
            ),
            Env(
                name=f"{LaunchConfigSettings.model_config['env_prefix']}max_restarts",  # noqa: E501
                value=1,
            ),
            # torch's LaunchConfig's default of 'spawn' doesnt seem to work
            # inside the argo emissary runtime context for some reason
            Env(
                name=f"{LaunchConfigSettings.model_config['env_prefix']}start_method",  # noqa: E501
                value="fork",
            ),
            Env(
                name=f"{LaunchConfigSettings.model_config['env_prefix']}rdzv_backend",  # noqa: E501
                value="static",
            ),
            Env(
                name=f"{LaunchConfigSettings.model_config['env_prefix']}rdzv_endpoint_url",  # noqa: E501
                value=f"{self.name}-{{{{workflow.uid}}}}.{self.k8s_namespace}.svc.cluster.local",  # noqa: E501
            ),
            Env(
                name=f"{LaunchConfigSettings.model_config['env_prefix']}rdzv_endpoint_port",  # noqa: E501
                value=DDP_PORT_NUMBER,
            ),
            Env(
                name=f"{LaunchConfigSettings.model_config['env_prefix']}run_id",  # noqa: E501
                value=1,
            ),
            Env(
                name=f"{LaunchConfigSettings.model_config['env_prefix']}tee",
                value=0,
            ),
        ]

        script_decorator_kwargs["name"] = f"{self.base_name}-{torch_node_rank}"

        labels: Dict = script_decorator_kwargs.get("labels", {})
        labels.update(
            {
                "torch-node": torch_node_rank,
                "torch-job": self.name,
            }
        )
        script_decorator_kwargs["labels"] = labels

        return script_decorator_kwargs

    def build_hera_task_factory(self) -> Callable[..., Task]:
        """Generates the task factory task_wrapper callable from the
        hera.workflows.script decorator definition. Needs to be called outide
        of an active hera context.

        Returns:
            List[Callable]: A list of callables which, if called inside an
                active hera DAG context, generate the hera Tasks, one for each
                node in the distributed torch run.
        """

        # add torch run environment variables to script kwargs
        task_factory = []

        for torch_node_rank in range(self.n_nodes):

            script_decorator_kwargs = self.build_script_decorator_kwargs(
                torch_node_rank
            )

            # this will invoke our custom TorchComponentInlineScriptRunner
            # under the hood
            script_wrapper = bettmensch_ai_script(
                script_class=self.script,
                **script_decorator_kwargs,
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


def as_torch_ddp_component(func: Callable) -> Callable[..., TorchDDPComponent]:
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

    def torch_ddp_component_factory(
        name: str = "",
        hera_template_kwargs: Dict = {},
        n_nodes: int = 1,
        min_nodes: int = None,
        nproc_per_node: int = 1,
        **component_inputs_kwargs,
    ) -> TorchDDPComponent:

        return TorchDDPComponent(
            func=func,
            name=name,
            hera_template_kwargs=hera_template_kwargs,
            n_nodes=n_nodes,
            min_nodes=min_nodes,
            nproc_per_node=nproc_per_node,
            **component_inputs_kwargs,
        )

    return torch_ddp_component_factory
