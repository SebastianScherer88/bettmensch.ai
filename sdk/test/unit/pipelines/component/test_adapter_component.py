from bettmensch_ai.pipelines.component import (
    AdapterInComponent,
    AdapterOutComponent,
)
from bettmensch_ai.pipelines.component.examples import (
    add_parameters_adapter_in_factory,
    add_parameters_adapter_out_factory,
    convert_to_artifact_adapter_in_factory,
    convert_to_artifact_adapter_out_factory,
    show_artifact_adapter_in_factory,
    show_artifact_adapter_out_factory,
)
from bettmensch_ai.pipelines.io import (
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
)
from bettmensch_ai.pipelines.pipeline_context import _pipeline_context
from hera.workflows import DAG, Parameter, WorkflowTemplate


def test_adapter_out_component___init__(test_function_and_task_inputs):
    """Tests of Component constructor."""

    test_function, test_task_inputs = test_function_and_task_inputs

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        test_component = AdapterOutComponent(
            func=test_function, name="test_name", **test_task_inputs
        )

    # validate addition of component to pipeline context
    assert test_component == _pipeline_context.components[0]

    # validate component attributes
    assert isinstance(test_component, AdapterOutComponent)
    assert test_component.implementation == "adapter_out"
    assert test_component.base_name == "test-name-adapter-out"
    assert test_component.name == "test-name-adapter-out-0"
    assert test_component.depends == "mock-component-0"

    # validate component template_outputs
    assert test_component.template_outputs["s3_prefix"].owner == test_component
    assert isinstance(
        test_component.template_outputs["s3_prefix"], OutputArtifact
    )  # noqa: E501


def test_adapter_in_component___init__(test_function_and_task_inputs):
    """Tests of Component constructor."""

    test_function, test_task_inputs = test_function_and_task_inputs

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        test_component = AdapterInComponent(
            func=test_function, name="test_name", **test_task_inputs
        )

    # validate addition of component to pipeline context
    assert test_component == _pipeline_context.components[0]

    # validate component attributes
    assert isinstance(test_component, AdapterInComponent)
    assert test_component.implementation == "adapter_in"
    assert test_component.base_name == "test-name-adapter-in"
    assert test_component.name == "test-name-adapter-in-0"
    assert test_component.depends == "mock-component-0"

    # validate component template_inputs
    assert list(test_component.template_inputs.keys()) == ["s3_prefix"]
    isinstance(test_component.template_inputs["s3_prefix"], InputArtifact)
    test_component.template_inputs["s3_prefix"].name = "s3_prefix"

    # validate component template_outputs
    assert test_component.template_outputs["a_out"].owner == test_component
    assert isinstance(
        test_component.template_outputs["a_out"], OutputParameter
    )  # noqa: E501
    assert test_component.template_outputs["b_out"].owner == test_component
    assert isinstance(test_component.template_outputs["b_out"], OutputArtifact)

    assert test_component.task_factory is None


def test_parameter_adapter_component_to_hera(
    test_output_dir, test_mock_pipeline
):
    """Declaration of Component using InputParameter and OutputParameter"""

    # mock active pipeline with 2 inputs
    pipeline_input_a = InputParameter(name="a", value=1)
    pipeline_input_a.set_owner(test_mock_pipeline)
    pipeline_input_b = InputParameter(name="b", value=2)
    pipeline_input_b.set_owner(test_mock_pipeline)

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        a_plus_b_out = add_parameters_adapter_out_factory(
            "a_plus_b",
            a=pipeline_input_a,
            b=pipeline_input_b,
        )

        a_plus_b_in = add_parameters_adapter_in_factory(
            "a_plus_b",
            a=InputParameter("a"),
            b=InputParameter("b"),
            s3_prefix=a_plus_b_out.outputs["s3_prefix"],
        )

        a_plus_b_plus_2_out = add_parameters_adapter_out_factory(
            "a_plus_b_plus_2",
            a=a_plus_b_in.outputs["sum"],
            b=InputParameter("two", 2),
        )

        a_plus_b_plus_2_in = add_parameters_adapter_in_factory(
            "a_plus_b_plus_2",
            a=InputParameter("a"),
            b=InputParameter("b"),
            s3_prefix=a_plus_b_plus_2_out.outputs["s3_prefix"],
        )

    a_plus_b_out.task_factory = a_plus_b_out.build_hera_task_factory()
    a_plus_b_in.task_factory = a_plus_b_in.build_hera_task_factory()
    a_plus_b_plus_2_out.task_factory = (
        a_plus_b_plus_2_out.build_hera_task_factory()
    )
    a_plus_b_plus_2_in.task_factory = (
        a_plus_b_plus_2_in.build_hera_task_factory()
    )

    with WorkflowTemplate(
        name="test-parameter-adapter-component-workflow-template",
        entrypoint="test_dag",
        namespace="argo",
        arguments=[
            Parameter(name="a", value=1),
            Parameter(name="b", value=2),
        ],
    ) as wft:

        with DAG(name="test_dag"):
            a_plus_b_out.to_hera()
            a_plus_b_in.to_hera()
            a_plus_b_plus_2_out.to_hera()
            a_plus_b_plus_2_in.to_hera()

    wft.to_file(test_output_dir)

    task_names = [task.name for task in wft.templates[0].tasks]
    assert task_names == [
        "a-plus-b-adapter-out-0",
        "a-plus-b-adapter-in-0",
        "a-plus-b-plus-2-adapter-out-0",
        "a-plus-b-plus-2-adapter-in-0",
    ]

    script_template_names = [template.name for template in wft.templates[1:]]
    assert script_template_names == [
        "a-plus-b-adapter-out",
        "a-plus-b-adapter-in",
        "a-plus-b-plus-2-adapter-out",
        "a-plus-b-plus-2-adapter-in",
    ]


def test_artifact_adapter_component_to_hera(
    test_output_dir,
    test_mock_pipeline,
):

    # mock active pipeline with 2 inputs
    pipeline_input_a = InputParameter(name="a", value=1)
    pipeline_input_a.set_owner(test_mock_pipeline)

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        convert_out = convert_to_artifact_adapter_out_factory(
            "convert_parameters",
            a=pipeline_input_a,
        )

        convert_in = convert_to_artifact_adapter_in_factory(
            "convert_parameters",
            a=InputParameter("a"),
            s3_prefix=convert_out.outputs["s3_prefix"],
        )

        show_out = show_artifact_adapter_out_factory(
            "show_artifacts",
            a=convert_in.outputs["a_art"],
        )

        show_in = show_artifact_adapter_in_factory(
            "show_artifacts",
            a=InputParameter("a"),
            s3_prefix=show_out.outputs["s3_prefix"],
        )

    convert_out.task_factory = convert_out.build_hera_task_factory()
    convert_in.task_factory = convert_in.build_hera_task_factory()
    show_out.task_factory = show_out.build_hera_task_factory()
    show_in.task_factory = show_in.build_hera_task_factory()

    with WorkflowTemplate(
        name="test-artifact-adapter-component-workflow-template",
        entrypoint="test_dag",
        namespace="argo",
        arguments=[
            Parameter(name="a", value=1),
        ],
    ) as wft:

        with DAG(name="test_dag"):
            convert_out.to_hera()
            convert_in.to_hera()
            show_out.to_hera()
            show_in.to_hera()

    wft.to_file(test_output_dir)

    task_names = [task.name for task in wft.templates[0].tasks]
    assert task_names == [
        "convert-parameters-adapter-out-0",
        "convert-parameters-adapter-in-0",
        "show-artifacts-adapter-out-0",
        "show-artifacts-adapter-in-0",
    ]

    script_template_names = [template.name for template in wft.templates[1:]]
    assert script_template_names == [
        "convert-parameters-adapter-out",
        "convert-parameters-adapter-in",
        "show-artifacts-adapter-out",
        "show-artifacts-adapter-in",
    ]
