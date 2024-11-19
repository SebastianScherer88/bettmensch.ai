from typing import Callable, Literal, Optional

from bettmensch_ai.pipelines.constants import (
    ARGO_NAMESPACE,
    COMPONENT_IMAGE,
    COMPONENT_IMPLEMENTATION,
    DDP_PORT_NAME,
    DDP_PORT_NUMBER,
)
from hera.workflows import Resource
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.distributed.elastic.multiprocessing.api import DefaultLogsSpecs
from torch.distributed.launcher.api import LaunchConfig, elastic_launch


class LaunchConfigSettings(BaseSettings):
    """The torch LaunchConfig fields as well as some additional fields handled
    by torch's command line utilities that need to be exposed to environment
    variables to allow configuration via a pod manifest's `env` field.

    Fields:
        min_nodes: Minimum amount of nodes that the user function will
                        be launched on. Elastic agent ensures that the user
                        function start only when the min_nodes amount enters
                        the rendezvous.
        max_nodes: Maximum amount of nodes that the user function
                        will be launched on.
        node_rank: The rank of the pod/node in the distributed torch run.
        nproc_per_node: On each node the elastic agent will launch
                            this amount of workers that will execute user
                            defined function.
        rdzv_backend: rdzv_backend to use in the rendezvous (zeus-adapter,
            etcd). We use "static" as that was successfully tested on K8s.
        rdzv_endpoint_url: The endpoint url of the rdzv sync. storage. This
            will be the DNS of the service that we route to the master node pod
            in K8s.
        rdzv_endpoint_port: The port of the endpoint of the rdvz sync. storage.
            We use the default 29500.
        run_id: The unique run id of the job (if not passed a unique one will
            be deduced from run environment - flow workflow id in flow - or
            auto generated).
        role: User defined role of the worker (defaults to "trainer").
        max_restarts: The maximum amount of restarts that elastic agent will
            conduct on workers before failure.
        monitor_interval: The interval in seconds that is used by the
            elastic_agent as a period of monitoring workers.
        log_dir: base log directory where log files are written. If not set,
                one is created in a tmp dir but NOT removed on exit.
        log_line_prefix_template: Not explained in torch docs, but passed to
            the LocalElasticAgent instance during distributed run
            orchestration.
        tee: configuration to "tee" stdout/stderr to console + log file.
    """

    min_nodes: int = 1
    max_nodes: int = 1
    node_rank: int = 0
    nproc_per_node: int = 1
    start_method: Literal[
        "spawn", "fork", "forkserver"
    ] = "fork"  # torch's LaunchConfig's default of 'spawn' doesnt seem to work
    # inside the argo emissary runtime context for some reason
    rdzv_backend: str = "static"
    rdzv_endpoint_url: str = "localhost"
    rdzv_endpoint_port: int = 29500
    run_id: str = ""
    role: str = ""
    max_restarts: int = 3
    monitor_interval: float = 30
    # redirects: Std = Std
    # tee: Std = Std
    log_dir: Optional[str] = None
    log_line_prefix_template: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="bettmensch_ai_torch_ddp_")


def get_launch_config(**config_settings_kwargs) -> LaunchConfig:
    """
    Utility torch distributed config constructor from environment variables
    using the LaunchConfigSettings.
    """

    launch_config_settings_from_env = LaunchConfigSettings(
        **config_settings_kwargs
    )

    print(
        f"""Torch distributed launch config settings: {
            launch_config_settings_from_env.model_dump()
        }"""
    )

    return LaunchConfig(
        min_nodes=launch_config_settings_from_env.min_nodes,
        max_nodes=launch_config_settings_from_env.max_nodes,
        nproc_per_node=launch_config_settings_from_env.nproc_per_node,
        logs_specs=DefaultLogsSpecs(
            log_dir=launch_config_settings_from_env.log_dir,
            #  redirects=launch_config_settings_from_env.redirects,
            #  tee=launch_config_settings_from_env.tee
        ),
        start_method=launch_config_settings_from_env.start_method,
        rdzv_endpoint=f"{launch_config_settings_from_env.rdzv_endpoint_url}:{launch_config_settings_from_env.rdzv_endpoint_port}",  # noqa: E501
        rdzv_backend=launch_config_settings_from_env.rdzv_backend,
        run_id=launch_config_settings_from_env.run_id,
        role=launch_config_settings_from_env.role,
        max_restarts=launch_config_settings_from_env.max_restarts,
        monitor_interval=launch_config_settings_from_env.monitor_interval,
        rdzv_configs={"rank": launch_config_settings_from_env.node_rank},
    )


class LaunchContext(BaseSettings):
    """Utility to grab the DDP environment variables set by torchrun as per
    https://pytorch.org/docs/stable/elastic/run.html#environment-variables.
    Useful to get information from within the worker process about the devices
    available to it."""

    # local process rank & local size (i.e. within node)
    local_rank: int
    local_world_size: int  # equivalent to nproc_per_node

    # global process rank & global size (i.e. across all nodes)
    rank: int
    world_size: int

    # global, role scoped process rank & global, role scoped size
    role_rank: int
    role_world_size: int

    # (global) node rank
    group_rank: int  # same as node_rank in the LaunchConfigSettings


def torch_ddp(**config_settings_kwargs):
    """Keyword decorator that wraps a callable in a torch distributed elastic
    launch runtime context.

    Example:

    @torch_distribute(run_id="test_id",min_nodes=1,max_nodes=2,nproc_per_node=1)
    def test_function(a: int = 1):
        return a

    # launch a torch.distributed elastic_launch
    test_function()
    """

    def decorator(function: Callable):
        def wrapper(*function_args):

            launch_config = get_launch_config(**config_settings_kwargs)

            elastic_launch(
                config=launch_config,
                entrypoint=function,
            )(*function_args)

        return wrapper

    return decorator


def get_jobset_resource_template(
    base_name: str,
    task_name: str,
    n_nodes: int,
    min_nodes: int,
    max_nodes: int,
    n_proc_per_node: int,
    cpu: str,
    memory: str,
    n_max_restarts: int = 1,
    start_method: str = "spawn",
    rdzv_backend: str = "static",
    image: str = COMPONENT_IMAGE.torch.value,
    namespace: str = ARGO_NAMESPACE,
) -> Resource:

    return Resource(
        name=f"{base_name}-create-{COMPONENT_IMPLEMENTATION.torch_ddp.value}-jobset",  # noqa: E501
        action="create",
        manifest=f"""apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: {task_name}-{{{{workflow.uid}}}}
  labels:
    kueue.x-k8s.io/queue-name: user-queue
  namespace: {namespace}
spec:
  network:
    enableDNSHostnames: true
    subdomain: torch-ddp-svc
  replicatedJobs:
    - name: {task_name}-{{{{workflow.uid}}}}
      replicas: 1
      template:
        spec:
          parallelism: {n_nodes}
          completions: {n_nodes}
          backoffLimit: 0
          template:
            spec:
              containers:
                - command:
                  - bash
                  - -c
                  - torchrun --rdzv_id=123 --nnodes={n_nodes} --nproc_per_node={n_proc_per_node} --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$JOB_COMPLETION_INDEX --role '' ./torch_ddp_test.py # noqa: E501
                  env:
                  - name: NCCL_DEBUG
                    value: 'INFO'
                  - name: {LaunchConfigSettings.model_config['env_prefix']}min_nodes # noqa: E501
                    value: '{min_nodes}'
                  - name: {LaunchConfigSettings.model_config['env_prefix']}max_nodes # noqa: E501
                    value: '{max_nodes}'
                  - name: {LaunchConfigSettings.model_config['env_prefix']}nproc_per_node # noqa: E501
                    value: '{n_proc_per_node}'
                  - name: {LaunchConfigSettings.model_config['env_prefix']}n_nodes # noqa: E501
                    value: '{n_nodes}'
                  - name: {LaunchConfigSettings.model_config['env_prefix']}node_rank # noqa: E501
                    valueFrom:
                      fieldRef:
                        fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index'] # noqa: E501
                  - name: {LaunchConfigSettings.model_config['env_prefix']}max_restarts # noqa: E501
                    value: '{n_max_restarts}'
                  - name: {LaunchConfigSettings.model_config['env_prefix']}start_method # noqa: E501
                    value: '{start_method}'
                  - name: {LaunchConfigSettings.model_config['env_prefix']}rdzv_backend # noqa: E501
                    value: '{rdzv_backend}'
                  - name: {LaunchConfigSettings.model_config['env_prefix']}rdzv_endpoint_url # noqa: E501
                    value: {task_name}-{{{{workflow.uid}}}}-{task_name}-{{{{workflow.uid}}}}-0-0.torch-ddp-svc # noqa: E501
                  - name: {LaunchConfigSettings.model_config['env_prefix']}rdzv_endpoint_port # noqa: E501
                    value: '{DDP_PORT_NUMBER}'
                  - name: {LaunchConfigSettings.model_config['env_prefix']}run_id # noqa: E501
                    value: '1'
                  - name: {LaunchConfigSettings.model_config['env_prefix']}tee # noqa: E501
                    value: '0'
                  image: {image}
                  name: torch-ddp
                  ports:
                  - containerPort: {DDP_PORT_NUMBER}
                    name: c10d
                    protocol: TCP
                  resources:
                    limits:
                      cpu: {cpu}
                      memory: {memory}
                    requests:
                      cpu: {cpu}
                      memory: {memory}
              restartPolicy: Never
        """,
    )


def get_configmap_resource_template():
    pass


def create_torch_ddp_service_template(
    component_base_name: str,
    component_task_name: str,
    namespace: str = ARGO_NAMESPACE,  # noqa: E501
) -> Resource:
    """Utility for a template creating the service resource required for
    accessing a TorchDDPComponent's master node on K8s."""

    return Resource(
        name=f"{component_base_name}-create-{COMPONENT_IMPLEMENTATION.torch_ddp.value}-service",  # noqa: E501
        action="create",
        manifest=f"""apiVersion: v1
kind: Service
metadata:
  name: {component_task_name}-{{{{workflow.uid}}}}
  namespace: {namespace}
  labels:
    workflows.argoproj.io/workflow: {{{{workflow.name}}}}
    torch-job: {component_task_name}
spec:
  clusterIP: None  # ClusterIP set to None for headless service.
  ports:
  - name: {DDP_PORT_NAME}  # Port for torchrun master<->worker node coms.
    port: {DDP_PORT_NUMBER}
    targetPort: {DDP_PORT_NUMBER}
  selector:
    workflows.argoproj.io/workflow: {{{{workflow.name}}}}
    torch-job: {component_task_name}
    torch-node: '0'  # Selector for pods associated with this service.
""",
    )


def delete_torch_ddp_service_template(
    component_base_name: str,
    component_task_name: str,
    namespace: str = ARGO_NAMESPACE,  # noqa: E501
) -> Resource:
    """Utility for a template deleting the service resource required for
    accessing a TorchDDPComponent's master node on K8s."""

    return Resource(
        name=f"{component_base_name}-delete-{COMPONENT_IMPLEMENTATION.torch_ddp.value}-service",  # noqa: E501
        action="delete",
        flags=[
            "service",
            "--selector",
            f"torch-job={component_task_name},workflows.argoproj.io/workflow={{{{workflow.name}}}}",  # noqa: E501
            "-n",
            namespace,
        ],
    )
