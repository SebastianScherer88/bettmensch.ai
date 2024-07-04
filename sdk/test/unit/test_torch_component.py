from bettmensch_ai import (
    COMPONENT_TYPE,
    PIPELINE_TYPE,
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
    TorchComponent,
    _pipeline_context,
    torch_component,
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

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        test_component = TorchComponent(
            func=test_function,
            name="test_name",
            n_nodes=2,
            min_nodes=2,
            max_nodes=4,
            nproc_per_node=2,
            **task_inputs
        )

    # validate addition of component to pipeline context
    assert test_component == _pipeline_context.components[0]

    # validate component attributes
    assert isinstance(test_component, TorchComponent)
    assert test_component.base_name == "test-name"
    assert test_component.name == "test-name-0"
    assert test_component.func == test_function
    assert test_component.hera_template_kwargs == {}
    assert test_component.n_nodes == 2
    assert test_component.min_nodes == 2
    assert test_component.max_nodes == 4
    assert test_component.nproc_per_node == 2
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


def test_torch_component_decorator(test_mock_pipeline, test_mock_component):
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

    test_component_factory = torch_component(test_function)

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

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        test_component = test_component_factory(
            n_nodes=2,
            min_nodes=2,
            max_nodes=4,
            nproc_per_node=2,
            name="test_name",
            **task_inputs
        )

    # validate addition of component to pipeline context
    assert test_component == _pipeline_context.components[0]

    # validate component attributes
    assert isinstance(test_component, TorchComponent)
    assert test_component.base_name == "test-name"
    assert test_component.name == "test-name-0"
    assert test_component.func == test_function
    assert test_component.hera_template_kwargs == {}
    assert test_component.n_nodes == 2
    assert test_component.min_nodes == 2
    assert test_component.max_nodes == 4
    assert test_component.nproc_per_node == 2

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


def test_parameter_torch_component_to_hera(
    test_add_function, test_mock_pipeline
):
    """Declaration of Component using InputParameter and OutputParameter"""

    add_component_factory = torch_component(test_add_function)

    # mock active pipeline with 2 inputs
    pipeline_input_a = InputParameter(name="a", value=1)
    pipeline_input_a.set_owner(test_mock_pipeline)
    pipeline_input_b = InputParameter(name="b", value=2)
    pipeline_input_b.set_owner(test_mock_pipeline)

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        a_plus_b = add_component_factory(
            "a_plus_b",
            n_nodes=2,
            min_nodes=1,
            max_nodes=4,
            nproc_per_node=5,
            a=pipeline_input_a,
            b=pipeline_input_b,
        )

        a_plus_b_plus_2 = add_component_factory(
            "a_plus_b_plus_2",
            n_nodes=2,
            min_nodes=2,
            max_nodes=3,
            nproc_per_node=4,
            a=a_plus_b.outputs["sum"],
            b=InputParameter("two", 2),
        )

    with WorkflowTemplate(
        name="test-parameter-component-workflow-template",
        entrypoint="test_dag",
        namespace="argo",
        arguments=[
            Parameter(name="a", value=1),
            Parameter(name="b", value=2),
        ],
    ) as wft:

        a_plus_b.service_templates = a_plus_b.build_service_templates()
        a_plus_b_plus_2.service_templates = (
            a_plus_b_plus_2.build_service_templates()
        )

        with DAG(name="test_dag"):
            a_plus_b.to_hera()
            a_plus_b_plus_2.to_hera()

    task_names = [task.name for task in wft.templates[4].tasks]
    assert task_names == [
        "a-plus-b-create-torch-service",
        "a-plus-b-0",
        "a-plus-b-0-worker-1",
        "a-plus-b-delete-torch-service",
        "a-plus-b-plus-2-create-torch-service",
        "a-plus-b-plus-2-0",
        "a-plus-b-plus-2-0-worker-1",
        "a-plus-b-plus-2-delete-torch-service",
    ]

    script_template_names = [template.name for template in wft.templates]
    assert script_template_names == [
        "a-plus-b-create-torch-service",
        "a-plus-b-delete-torch-service",
        "a-plus-b-plus-2-create-torch-service",
        "a-plus-b-plus-2-delete-torch-service",
        "test_dag",
        "a-plus-b-0",
        "a-plus-b-1",
        "a-plus-b-plus-2-0",
        "a-plus-b-plus-2-1",
    ]

    assert wft.templates[5].labels["torch-node"] == "0"
    assert wft.templates[5].labels["app"].startswith("a-plus-b-0-")
    assert wft.templates[6].labels["torch-node"] == "1"

    assert wft.templates[7].labels["torch-node"] == "0"
    assert wft.templates[7].labels["app"].startswith("a-plus-b-plus-2-0-")
    assert wft.templates[8].labels["torch-node"] == "1"


def test_artifact_torch_component_to_hera(
    test_convert_to_artifact_function,
    test_show_artifact_function,
    test_mock_pipeline,
):

    convert_component_factory = torch_component(
        test_convert_to_artifact_function
    )
    show_component_factory = torch_component(test_show_artifact_function)

    # mock active pipeline with 2 inputs
    pipeline_input_a = InputParameter(name="a", value=1)
    pipeline_input_a.set_owner(test_mock_pipeline)
    pipeline_input_b = InputParameter(name="b", value=2)
    pipeline_input_b.set_owner(test_mock_pipeline)

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        convert = convert_component_factory(
            "convert_parameters",
            n_nodes=2,
            min_nodes=1,
            nproc_per_node=5,
            a=pipeline_input_a,
            b=pipeline_input_b,
        )

        show = show_component_factory(
            "show_artifacts",
            a=convert.outputs["a_art"],
            b=convert.outputs["b_art"],
        )

    with WorkflowTemplate(
        name="test-artifact-component-workflow-template",
        entrypoint="test_dag",
        namespace="argo",
        arguments=[
            Parameter(name="a", value=1),
            Parameter(name="b", value=2),
        ],
    ) as wft:

        convert.service_templates = convert.build_service_templates()
        show.service_templates = show.build_service_templates()

        with DAG(name="test_dag"):
            convert.to_hera()
            show.to_hera()

    task_names = [task.name for task in wft.templates[4].tasks]
    assert task_names == [
        "convert-parameters-create-torch-service",
        "convert-parameters-0",
        "convert-parameters-0-worker-1",
        "convert-parameters-delete-torch-service",
        "show-artifacts-create-torch-service",
        "show-artifacts-0",
        "show-artifacts-delete-torch-service",
    ]

    script_template_names = [template.name for template in wft.templates]
    assert script_template_names == [
        "convert-parameters-create-torch-service",
        "convert-parameters-delete-torch-service",
        "show-artifacts-create-torch-service",
        "show-artifacts-delete-torch-service",
        "test_dag",
        "convert-parameters-0",
        "convert-parameters-1",
        "show-artifacts-0",
    ]

    assert wft.templates[5].labels["torch-node"] == "0"
    assert wft.templates[5].labels["app"].startswith("convert-parameters-0-")
    assert wft.templates[6].labels["torch-node"] == "1"

    assert wft.templates[7].labels["torch-node"] == "0"
    assert wft.templates[7].labels["app"].startswith("show-artifacts-0-")
