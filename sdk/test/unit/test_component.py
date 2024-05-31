from bettmensch_ai import (
    COMPONENT_TYPE,
    PIPELINE_TYPE,
    Component,
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
    _pipeline_context,
    component,
)
from hera.workflows import DAG, Parameter, WorkflowTemplate


def test_component___init__(test_mock_pipeline, test_mock_component):
    """Tests of Component constructor."""

    def test_function(
        a: InputParameter,
        b: InputParameter,
        c: InputParameter,
        d: InputArtifact,
        a_out: OutputParameter,
        b_out: OutputArtifact,
    ):
        pass

    _pipeline_context.activate()
    _pipeline_context.clear()

    test_input_a = InputParameter("fixed", 1)
    test_input_b = InputParameter("mock_pipe_in", 1)
    test_input_b.set_owner(test_mock_pipeline)
    test_input_c = OutputParameter("mock_comp_out_param")
    test_input_c.set_owner(test_mock_component)
    test_input_d = OutputArtifact("mock_comp_out_art")
    test_input_d.set_owner(test_mock_component)

    task_inputs = {
        "a": test_input_a,
        "b": test_input_b,
        "c": test_input_c,
        "d": test_input_d,
    }

    # add components to pipeline context
    test_component = Component(
        func=test_function, name="test_name", **task_inputs
    )

    # close pipeline context
    _pipeline_context.deactivate()

    # validate addition of component to pipeline context
    assert test_component == _pipeline_context.components[0]

    # validate component attributes
    assert isinstance(test_component, Component)
    assert test_component.base_name == "test-name"
    assert test_component.name == "test-name-0"
    assert test_component.func == test_function
    assert test_component.hera_template_kwargs == {}
    assert test_component.depends == "mock-component-0"

    # validate component task_inputs
    for task_input_name in ("a", "b", "c", "d"):
        assert (
            test_component.task_inputs[task_input_name].name == task_input_name
        )
        assert (
            test_component.task_inputs[task_input_name].owner == test_component
        )
        assert (
            test_component.task_inputs[task_input_name].source
            is task_inputs[task_input_name]
        )

    assert test_component.task_inputs["a"].value == task_inputs["a"].value
    assert test_component.task_inputs["b"].value == task_inputs["b"].value
    assert test_component.task_inputs["c"].value is None

    # validate component template_inputs
    assert list(test_component.template_inputs.keys()) == ["d"]
    isinstance(test_component.template_inputs["d"], InputArtifact)
    test_component.template_inputs["d"].name = "d"

    # validate component template_outputs
    assert test_component.template_outputs["a_out"].owner == test_component
    assert isinstance(test_component.template_outputs["a_out"], OutputParameter)
    assert test_component.template_outputs["b_out"].owner == test_component
    assert isinstance(test_component.template_outputs["b_out"], OutputArtifact)

    assert test_component.task_factory is not None


def test_component_decorator(test_mock_pipeline, test_mock_component):
    """Tests of Component constructor."""

    def test_function(
        a: InputParameter,
        b: InputParameter,
        c: InputParameter,
        d: InputArtifact,
        a_out: OutputParameter,
        b_out: OutputArtifact,
    ):
        pass

    test_component_factory = component(test_function)

    _pipeline_context.activate()
    _pipeline_context.clear()

    test_input_a = InputParameter("fixed", 1)
    test_input_b = InputParameter("mock_pipe_in", 1)
    test_input_b.set_owner(test_mock_pipeline)
    test_input_c = OutputParameter("mock_comp_out_param")
    test_input_c.set_owner(test_mock_component)
    test_input_d = OutputArtifact("mock_comp_out_art")
    test_input_d.set_owner(test_mock_component)

    task_inputs = {
        "a": test_input_a,
        "b": test_input_b,
        "c": test_input_c,
        "d": test_input_d,
    }

    # add components to pipeline context
    test_component = test_component_factory(name="test_name", **task_inputs)

    # close pipeline context
    _pipeline_context.deactivate()

    # validate addition of component to pipeline context
    assert test_component == _pipeline_context.components[0]

    # validate component attributes
    assert isinstance(test_component, Component)
    assert test_component.base_name == "test-name"
    assert test_component.name == "test-name-0"
    assert test_component.func == test_function
    assert test_component.hera_template_kwargs == {}

    # validate component task_inputs
    for task_input_name in ("a", "b", "c", "d"):
        assert (
            test_component.task_inputs[task_input_name].name == task_input_name
        )
        assert (
            test_component.task_inputs[task_input_name].owner == test_component
        )
        assert (
            test_component.task_inputs[task_input_name].source
            is task_inputs[task_input_name]
        )

    assert test_component.task_inputs["a"].value == task_inputs["a"].value
    assert test_component.task_inputs["b"].value == task_inputs["b"].value
    assert test_component.task_inputs["c"].value is None

    # validate component template_inputs
    assert list(test_component.template_inputs.keys()) == ["d"]
    isinstance(test_component.template_inputs["d"], InputArtifact)
    test_component.template_inputs["d"].name = "d"

    # validate component template_outputs
    assert test_component.template_outputs["a_out"].owner == test_component
    assert isinstance(test_component.template_outputs["a_out"], OutputParameter)
    assert test_component.template_outputs["b_out"].owner == test_component
    assert isinstance(test_component.template_outputs["b_out"], OutputArtifact)

    assert test_component.task_factory is not None


def test_component_to_hera(test_add_function, test_mock_pipeline):
    """Declaration of Component using InputParameter and OutputParameter"""

    add_component_factory = component(test_add_function)

    # mock active pipeline with 3 inputs
    pipeline_input_a = InputParameter(name="a", value=1)
    pipeline_input_a.set_owner(test_mock_pipeline)
    pipeline_input_b = InputParameter(name="b", value=2)
    pipeline_input_b.set_owner(test_mock_pipeline)

    _pipeline_context.activate()
    _pipeline_context.clear()

    # add components to pipeline context
    a_plus_b = add_component_factory(
        "a_plus_b",
        a=pipeline_input_a,
        b=pipeline_input_b,
    )

    a_plus_b_plus_2 = add_component_factory(
        "a_plus_b_plus_2", a=a_plus_b.outputs["sum"], b=InputParameter("two", 2)
    )

    # close pipeline context
    _pipeline_context.deactivate()

    with WorkflowTemplate(
        name="test-parameter-component-workflow-template",
        entrypoint="test_dag",
        namespace="argo",
        arguments=[
            Parameter(name="a", value=1),
            Parameter(name="b", value=2),
        ],
    ) as wft:

        with DAG(name="test_dag"):
            a_plus_b.to_hera()
            a_plus_b_plus_2.to_hera()

    task_names = [task.name for task in wft.templates[0].tasks]
    assert task_names == ["a-plus-b-0", "a-plus-b-plus-2-0"]

    script_template_names = [template.name for template in wft.templates[1:]]
    assert script_template_names == ["a-plus-b", "a-plus-b-plus-2"]
