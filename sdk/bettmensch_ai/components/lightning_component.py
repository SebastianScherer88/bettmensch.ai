from typing import Callable, Dict, Optional, Union

from bettmensch_ai.components.torch_component import TorchComponent
from bettmensch_ai.constants import COMPONENT_IMAGE, COMPONENT_IMPLEMENTATION


class LightningComponent(TorchComponent):

    implementation: str = COMPONENT_IMPLEMENTATION.lightning.value
    default_image: str = COMPONENT_IMAGE.lightning.value

    # if no resources are specified, set minimal requirements derived from
    # testing the ddp example on K8s
    cpu: Optional[Union[float, int, str]] = "700m"
    memory: Optional[str] = "1Gi"


def lightning_component(func: Callable) -> Callable[..., LightningComponent]:
    """Takes a calleable and generates a configured LightningComponent factory
    that will generate a LightningComponent version of the callable if invoked
    inside an active PipelineContext.

    Usage:

    ```python
    @bettmensch_ai.components.lightning_component #-> component factory
    def add(
        a: InputParameter = 1,
        b: InputParameter = 2,
        sum: OutputParameter = None
    ):
        sum.assign(a + b)
    ```

    Decorating the above `add` method returns a component factory that
    generates a LightningComponent class instance when called from within an
    active PipelineContext.
    """

    def lightning_component_factory(
        name: str = "",
        hera_template_kwargs: Dict = {},
        n_nodes: int = 1,
        min_nodes: int = None,
        nproc_per_node: int = 1,
        **component_inputs_kwargs,
    ) -> LightningComponent:

        return LightningComponent(
            func=func,
            name=name,
            hera_template_kwargs=hera_template_kwargs,
            n_nodes=n_nodes,
            min_nodes=min_nodes,
            nproc_per_node=nproc_per_node,
            **component_inputs_kwargs,
        )

    return lightning_component_factory
