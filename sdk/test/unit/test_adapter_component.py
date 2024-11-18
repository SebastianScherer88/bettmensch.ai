from bettmensch_ai.pipelines.component import (
    AdapterInComponent,
    AdapterOutComponent,
)
from bettmensch_ai.pipelines.io import (
    InputParameter,
    OutputArtifact,
    OutputParameter,
)
from bettmensch_ai.pipelines.pipeline import _pipeline_context


def test_adapter_out_component___init__(test_function_and_task_inputs):
    """Tests of Component constructor."""

    test_function, test_task_inputs = test_function_and_task_inputs

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        test_component = AdapterOutComponent(
            func=test_function, **test_task_inputs
        )

    # validate addition of component to pipeline context
    assert test_component == _pipeline_context.components[0]

    # validate component attributes
    assert isinstance(test_component, AdapterOutComponent)
    assert test_component.implementation == "adapter_out"
    assert test_component.base_name == "adapter-out"
    assert test_component.name == "adapter-out-0"
    assert test_component.depends == "mock-component-0"

    # validate component template_outputs
    assert test_component.template_outputs["s3_prefix"].owner == test_component
    assert isinstance(
        test_component.template_outputs["s3_prefix"], OutputParameter
    )  # noqa: E501


def test_adapter_in_component___init__(test_function_and_task_inputs):
    """Tests of Component constructor."""

    test_function, test_task_inputs = test_function_and_task_inputs

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        test_component = AdapterInComponent(
            func=test_function, **test_task_inputs
        )

    # validate addition of component to pipeline context
    assert test_component == _pipeline_context.components[0]

    # validate component attributes
    assert isinstance(test_component, AdapterInComponent)
    assert test_component.implementation == "adapter_in"
    assert test_component.base_name == "adapter-in"
    assert test_component.name == "adapter-in-0"
    assert test_component.depends == "mock-component-0"

    # validate component template_inputs
    assert test_component.template_inputs["s3_prefix"].owner == test_component
    assert isinstance(
        test_component.template_outputs["s3_prefix"], InputParameter
    )  # noqa: E501

    # validate component template_outputs
    assert test_component.template_outputs["a_out"].owner == test_component
    assert isinstance(
        test_component.template_outputs["a_out"], OutputParameter
    )  # noqa: E501
    assert test_component.template_outputs["b_out"].owner == test_component
    assert isinstance(test_component.template_outputs["b_out"], OutputArtifact)

    assert test_component.task_factory is None
