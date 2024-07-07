from typing import Callable, Optional
from uuid import uuid4

from hera.workflows import Resource
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.distributed.elastic.multiprocessing.api import Std
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
        rdzv_endpoint_url: The endpoint url of the rdzv sync. storage. This will
            be the DNS of the service that we route to the master node pod in
            K8s.
        rdzv_endpoint_port: The port of the endpoint of the rdvz sync. storage.
            We use the default 29500.
        run_id: The unique run id of the job (if not passed a unique one will be
                deduced from run environment - flow workflow id in flow - or auto generated).
        role: User defined role of the worker (defaults to "trainer").
        max_restarts: The maximum amount of restarts that elastic agent will conduct
                    on workers before failure.
        monitor_interval: The interval in seconds that is used by the elastic_agent
                        as a period of monitoring workers.
        log_dir: base log directory where log files are written. If not set,
                one is created in a tmp dir but NOT removed on exit.
        log_line_prefix_template: Not explained in torch docs, but passed to the
            LocalElasticAgent instance during distributed run orchestration.
        tee: configuration to "tee" stdout/stderr to console + log file.
    """

    min_nodes: int = 1
    max_nodes: int = 1
    node_rank: int = 0
    nproc_per_node: int = 1
    rdzv_backend: str = "static"
    rdvz_endpoint_url: str = "localhost"
    rdvz_endpoint_port: int = 29500
    run_id: str = ""
    role: str = ""
    max_restarts: int = 3
    monitor_interval: float = 30
    redirects: str = "3"
    tee: str = "0"
    log_dir: Optional[str] = None
    log_line_prefix_template: Optional[str] = None

    model_config = SettingsConfigDict(
        env_prefix="bettmensch_ai_distributed_torch_"
    )


def get_launch_config(**config_settings_kwargs) -> LaunchConfig:
    """
    Utility torch distributed config constructor from environment variables
    using the LaunchConfigSettings.
    """

    launch_config_settings_from_env = LaunchConfigSettings(
        **config_settings_kwargs
    )

    print(
        f"Torch distributed launch config settings: {launch_config_settings_from_env.model_dump()}"
    )

    return LaunchConfig(
        min_nodes=launch_config_settings_from_env.min_nodes,
        max_nodes=launch_config_settings_from_env.max_nodes,
        nproc_per_node=launch_config_settings_from_env.nproc_per_node,
        rdzv_endpoint=f"{launch_config_settings_from_env.rdvz_endpoint_url}:{launch_config_settings_from_env.rdvz_endpoint_port}",
        rdzv_backend=launch_config_settings_from_env.rdzv_backend,
        run_id=launch_config_settings_from_env.run_id,
        role=launch_config_settings_from_env.role,
        max_restarts=launch_config_settings_from_env.max_restarts,
        monitor_interval=launch_config_settings_from_env.monitor_interval,
        # tee=Std.from_str(launch_config_settings_from_env.tee),
        # redirects=Std.from_str(launch_config_settings_from_env.redirects),
        # log_dir=launch_config_settings_from_env.log_dir,
        # log_line_prefix_template=launch_config_settings_from_env.log_line_prefix_template,
        rdzv_configs={"rank": launch_config_settings_from_env.node_rank},
    )


def torch_distribute(**config_settings_kwargs):
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

            # provide a consistent run_id naming convention as a convenience
            if "run_id" not in config_settings_kwargs.keys():
                config_settings_kwargs[
                    "run_id"
                ] = f"{function.__name__}_{str(uuid4())}"

            launch_config = get_launch_config(**config_settings_kwargs)

            elastic_launch(
                config=launch_config,
                entrypoint=function,
            )(*function_args)

        return wrapper

    return decorator


def create_torch_service_template(
    component_base_name: str, service_name: str, namespace: str = "argo"
) -> Resource:
    """Utility for a template creating the service resource required for
    accessing a TorchComponent's master node on K8s."""

    return Resource(
        name=f"{component_base_name}-create-torch-service",
        action="create",
        manifest=f"""apiVersion: v1
kind: Service
metadata:
  name: {service_name}
  namespace: {namespace}
  labels:
    app: {service_name}
spec:
  clusterIP: None  # ClusterIP set to None for headless service.
  ports:
  - name: nccl  # Port for torchrun master-worker node communication.
    port: 29500
    targetPort: 29500
  selector:
    app: {service_name}
    torch-node: '0'  # Selector for pods associated with this service.
""",
    )


def delete_torch_service_template(
    component_base_name: str, service_name: str, namespace: str = "argo"
) -> Resource:
    """Utility for a template deleting the service resource required for
    accessing a TorchComponent's master node on K8s."""

    return Resource(
        name=f"{component_base_name}-delete-torch-service",
        action="delete",
        flags=["service", "--selector", f"app={service_name}", "-n", namespace],
    )
