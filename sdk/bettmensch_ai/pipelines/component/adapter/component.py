from typing import Any, Callable, Dict, List, Type, Union

from bettmensch_ai.pipelines.component.base import BettmenschAIBaseScript
from bettmensch_ai.pipelines.component.standard import Component
from bettmensch_ai.pipelines.constants import (
    COMPONENT_IMAGE,
    COMPONENT_IMPLEMENTATION,
)
from bettmensch_ai.pipelines.io import (
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
)

from .script import BettmenschAIAdapterInScript, BettmenschAIAdapterOutScript


class BaseAdapterComponent(Component):
    """A base class for the adapater type component classes."""

    default_image: COMPONENT_IMAGE.adapter.value


class AdapterOutComponent(BaseAdapterComponent):
    """Utility component for implementing components that run outside of the
    argo workflows context. Takes inputs and writes them to a
    designated S3 location where external processes can pick them up more
    easily. Logical inverse of the AdapterInComponent.
    """

    implementation: str = COMPONENT_IMPLEMENTATION.adapter_out.value
    script: Type[BettmenschAIBaseScript] = BettmenschAIAdapterOutScript

    def __init__(
        self,
        func: Callable,
        name: str,
        hera_template_kwargs: Dict = {},
        **component_inputs_kwargs: Dict[
            str, Union[InputParameter, OutputParameter, OutputArtifact]
        ],
    ):

        super().__init__(
            func,
            f"{name}-{COMPONENT_IMPLEMENTATION.adapter_out.value}",
            hera_template_kwargs,
            **component_inputs_kwargs,
        )
        # overwrite template_outputs attribute that was populated in previous
        # method call with fixed s3_prefix output
        output = OutputArtifact(name="s3_prefix")
        output.set_owner(self)

        self.template_outputs = {"s3_prefix": output}


class AdapterInComponent(BaseAdapterComponent):
    """Utility component for implementing components that run outside of the
    argo workflows context. Takes S3 parameter and artifact files produced by
    a AdapterOutComponent and converts them back to argo workflow outputs.
    Logical inverse of the AdapterOutComponent.
    """

    implementation: str = COMPONENT_IMPLEMENTATION.adapter_in.value
    non_function_inputs: str = ("s3_prefix",)
    script: Type[BettmenschAIBaseScript] = BettmenschAIAdapterInScript

    def __init__(
        self,
        func: Callable,
        name: str,
        hera_template_kwargs: Dict = {},
        **component_inputs_kwargs: Dict[
            str, Union[InputParameter, OutputParameter, OutputArtifact]
        ],
    ):

        super().__init__(
            func,
            f"{name}-{COMPONENT_IMPLEMENTATION.adapter_in.value}",
            hera_template_kwargs,
            **component_inputs_kwargs,
        )
        # overwrite template_inputs attribute that was populated in previous
        # method call with fixed s3 prefix input
        input = InputArtifact(name="s3_prefix")
        input.set_owner(self)

        self.template_inputs = {"s3_prefix": input}


def adapter_component(
    adapter_type: str,
    func: Callable,
) -> Callable[
    [str, Dict, List[Any]], Union[AdapterInComponent, AdapterOutComponent]
]:
    """Takes a calleable and generates either configured AdapterInComponent
    or AdapterOutComponent factory that will generate the respective component
    version of the callable if invoked inside an active PipelineContext.

    Usage:

    ```python
    def add(
        a: InputParameter = 1,
        b: InputParameter = 2,
        sum: OutputParameter = None
    ):
        sum.assign(a + b)

    from bettmensch_ai.pipelines import adapter_component
    add_adapter_out = adapter_component("adapter_out")(add)
    add_adapter_in = adapter_component("adapter_in")(add)
    ```

    Decorating the above `add` method returns a component factory that
    generates a Component class instance when called from within an active
    PipelineContext.
    """

    assert adapter_type in (
        COMPONENT_IMPLEMENTATION.adapter_in.value,
        COMPONENT_IMPLEMENTATION.adapter_out.value,
    ), f"Adapter type must be one of {(COMPONENT_IMPLEMENTATION.adapter_in.value, COMPONENT_IMPLEMENTATION.adapter_out.value)}"  # noqa: E501

    def adapter_component_factory(
        name: str = "",
        hera_template_kwargs: Dict = {},
        **component_inputs_kwargs,
    ) -> Union[AdapterInComponent, AdapterOutComponent]:

        if adapter_type == COMPONENT_IMPLEMENTATION.adapter_in.value:
            return AdapterInComponent(
                func=func,
                name=name,
                hera_template_kwargs=hera_template_kwargs,
                **component_inputs_kwargs,
            )
        elif adapter_type == COMPONENT_IMPLEMENTATION.adapter_out.value:
            return AdapterOutComponent(
                func=func,
                name=name,
                hera_template_kwargs=hera_template_kwargs,
                **component_inputs_kwargs,
            )

    return adapter_component_factory
