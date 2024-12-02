from bettmensch_ai.pipelines.component import Component, as_component
from bettmensch_ai.pipelines.component.examples import (
    add_parameters_factory,
    convert_to_artifact_factory,
    show_artifact_factory,
)
from bettmensch_ai.pipelines.io import (
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
)
from bettmensch_ai.pipelines.pipeline_context import _pipeline_context
from hera.workflows import DAG, Artifact, Parameter, WorkflowTemplate, models


def test_component___init__(test_function_and_task_inputs):
    """Tests of Component constructor."""

    test_function, test_task_inputs = test_function_and_task_inputs

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        test_component = (
            Component(func=test_function, name="test_name", **test_task_inputs)
            .set_cpu(0.5)
            .set_memory("100Mi")
            .set_gpus(1)
        )

    # --- validate addition of component to pipeline context
    assert test_component == _pipeline_context.components[0]

    # --- validate component instance's attributes
    assert isinstance(test_component, Component)
    assert test_component.implementation == "standard"
    assert test_component.base_name == "test-name"
    assert test_component.name == "test-name-0"
    assert test_component.func == test_function
    assert test_component.hera_template_kwargs == {}
    assert test_component.cpu == 0.5
    assert test_component.memory == "100Mi"
    assert test_component.gpus == 1
    assert test_component.ephemeral is None
    assert test_component.custom_resources is None
    assert test_component.depends == "mock-component-0"

    # --- validate component instance's io attributes
    # note that InputParameter type argument are automatically injected by
    # hera's script's `build_inputs` method, so arent being constructed here
    # explicitly
    assert test_component.template_inputs == {
        "d": InputArtifact(name="d").set_owner(test_component),
    }

    assert test_component.template_outputs == {
        "a_out": OutputParameter(name="a_out").set_owner(test_component),
        "b_out": OutputArtifact(name="b_out").set_owner(test_component),
    }

    assert test_component.task_inputs == {
        "a": InputParameter(name="a", value=1)
        .set_owner(test_component)
        .set_source(test_task_inputs["a"]),
        "b": InputParameter(name="b", value=1)
        .set_owner(test_component)
        .set_source(test_task_inputs["b"]),
        "c": InputParameter(name="c")
        .set_owner(test_component)
        .set_source(test_task_inputs["c"]),
        "d": InputArtifact(name="d")
        .set_owner(test_component)
        .set_source(test_task_inputs["d"]),
    }

    assert test_component.task_factory is None


def test_component_decorator(
    test_function_and_task_inputs,
):
    """Tests of Component constructor."""

    test_function, test_task_inputs = test_function_and_task_inputs

    test_component_factory = as_component(test_function)

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        test_component = (
            test_component_factory(name="test_name", **test_task_inputs)
            .set_cpu(0.5)
            .set_memory("100Mi")
            .set_gpus(1)
        )

    # validate addition of component to pipeline context
    assert test_component == _pipeline_context.components[0]

    # --- validate component instance's attributes
    assert isinstance(test_component, Component)
    assert test_component.implementation == "standard"
    assert test_component.base_name == "test-name"
    assert test_component.name == "test-name-0"
    assert test_component.func == test_function
    assert test_component.hera_template_kwargs == {}
    assert test_component.cpu == 0.5
    assert test_component.memory == "100Mi"
    assert test_component.gpus == 1
    assert test_component.ephemeral is None
    assert test_component.custom_resources is None
    assert test_component.depends == "mock-component-0"

    # --- validate component instance's io attributes
    # note that InputParameter type argument are automatically injected by
    # hera's script's `build_inputs` method, so arent being constructed here
    # explicitly
    assert test_component.template_inputs == {
        "d": InputArtifact(name="d").set_owner(test_component),
    }

    assert test_component.template_outputs == {
        "a_out": OutputParameter(name="a_out").set_owner(test_component),
        "b_out": OutputArtifact(name="b_out").set_owner(test_component),
    }

    assert test_component.task_inputs == {
        "a": InputParameter(name="a", value=1)
        .set_owner(test_component)
        .set_source(test_task_inputs["a"]),
        "b": InputParameter(name="b", value=1)
        .set_owner(test_component)
        .set_source(test_task_inputs["b"]),
        "c": InputParameter(name="c")
        .set_owner(test_component)
        .set_source(test_task_inputs["c"]),
        "d": InputArtifact(name="d")
        .set_owner(test_component)
        .set_source(test_task_inputs["d"]),
    }

    assert test_component.task_factory is None


def test_parameter_component_to_hera(test_output_dir, test_mock_pipeline):
    """Declaration of Component using InputParameter and OutputParameter"""

    # mock active pipeline with 2 inputs
    pipeline_input_a = InputParameter(name="a", value=1)
    pipeline_input_a.set_owner(test_mock_pipeline)
    pipeline_input_b = InputParameter(name="b", value=2)
    pipeline_input_b.set_owner(test_mock_pipeline)

    with _pipeline_context:
        _pipeline_context.clear()

        print(
            "test_parameter_component_to_hera pipeline context:"
            f"{_pipeline_context}"
        )

        # add components to pipeline context
        a_plus_b = (
            add_parameters_factory(
                "a_plus_b",
                a=pipeline_input_a,
                b=pipeline_input_b,
            )
            .set_cpu(1)
            .set_memory("1Gi")
        )

        a_plus_b_plus_2 = (
            add_parameters_factory(
                "a_plus_b_plus_2",
                a=a_plus_b.outputs["sum"],
                b=InputParameter("two", 2),
            )
            .set_gpus(1)
            .set_ephemeral("1Ti")
        )

    a_plus_b.task_factory = a_plus_b.build_hera_task_factory()
    a_plus_b_plus_2.task_factory = a_plus_b_plus_2.build_hera_task_factory()

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

    wft.to_file(test_output_dir)

    # --- validate `to_hera` outputs via argo.workflows.WorkflowTemplate
    # instance

    # validate tasks
    first_task = wft.templates[0].tasks[0]
    assert first_task.name == "a-plus-b-0"
    assert first_task.template.name == "a-plus-b"
    assert first_task.arguments == [
        Parameter(name="a", value="{{inputs.parameters.a}}"),
        Parameter(name="b", value="{{inputs.parameters.b}}"),
    ]
    assert first_task.depends == ""

    second_task = wft.templates[0].tasks[1]
    assert second_task.name == "a-plus-b-plus-2-0"
    assert second_task.template.name == "a-plus-b-plus-2"
    assert second_task.arguments == [
        Parameter(
            name="a", value="{{tasks.a-plus-b-0.outputs.parameters.sum}}"
        ),
        Parameter(name="b", value=2),
    ]
    assert second_task.depends == "a-plus-b-0"

    # validate script templates
    first_script_template = wft.templates[1]
    first_script_template.name == "a-plus-b"
    first_script_template.inputs == [
        Parameter(name="a", value=1),
        Parameter(name="b", value=2),
        Parameter(name="sum", value=None),
    ]
    first_script_template.outputs == [
        Parameter(name="sum", value_from=models.ValueFrom(path="sum")),
    ]

    second_script_template = wft.templates[2]
    second_script_template.name == "a-plus-b-plus-2"
    second_script_template.inputs == [
        Parameter(name="a", value=1),
        Parameter(name="b", value=2),
        Parameter(name="sum", value=None),
    ]
    second_script_template.outputs == [
        Parameter(name="sum", value_from=models.ValueFrom(path="sum")),
    ]


def test_artifact_component_to_hera(
    test_output_dir,
    test_mock_pipeline,
):

    # mock active pipeline with 2 inputs
    pipeline_input_a = InputParameter(name="a", value=1)
    pipeline_input_a.set_owner(test_mock_pipeline)

    with _pipeline_context:
        _pipeline_context.clear()

        print(
            "test_artifact_component_to_hera pipeline context:"
            f"{_pipeline_context}"
        )

        # add components to pipeline context
        convert = (
            convert_to_artifact_factory(
                "convert_parameters",
                a=pipeline_input_a,
            )
            .set_cpu(0.8)
            .set_memory("2Pi")
        )

        show = (
            show_artifact_factory(
                "show_artifacts",
                a=convert.outputs["a_art"],
            )
            .set_gpus(2)
            .set_ephemeral("10Ki")
        )

    convert.task_factory = convert.build_hera_task_factory()
    show.task_factory = show.build_hera_task_factory()

    with WorkflowTemplate(
        name="test-artifact-component-workflow-template",
        entrypoint="test_dag",
        namespace="argo",
        arguments=[
            Parameter(name="a", value=1),
        ],
    ) as wft:

        with DAG(name="test_dag"):
            convert.to_hera()
            show.to_hera()

    wft.to_file(test_output_dir)

    # --- validate `to_hera` outputs via argo.workflows.WorkflowTemplate
    # instance

    # validate tasks
    first_task = wft.templates[0].tasks[0]
    assert first_task.name == "convert-parameters-0"
    assert first_task.template.name == "convert-parameters"
    assert first_task.arguments == [
        Parameter(name="a", value="{{inputs.parameters.a}}"),
    ]
    assert first_task.depends == ""

    second_task = wft.templates[0].tasks[1]
    assert second_task.name == "show-artifacts-0"
    assert second_task.template.name == "show-artifacts"
    assert second_task.arguments == [
        Artifact(
            name="a",
            from_="{{tasks.convert-parameters-0.outputs.artifacts.a_art}}",
        ),
    ]
    assert second_task.depends == "convert-parameters-0"

    # validate script templates
    first_script_template = wft.templates[1]
    first_script_template.name == "convert-parameters"
    first_script_template.inputs == [
        Parameter(name="a"),
        Parameter(name="a_art", value=None),
    ]
    first_script_template.outputs == [
        Artifact(name="a_art", path="a_art"),
    ]

    second_script_template = wft.templates[2]
    second_script_template.name == "show-artifacts"
    second_script_template.inputs == [
        Artifact(name="a", path="a"),
        Parameter(name="b", value=None),
    ]
    second_script_template.outputs == [
        Artifact(name="b", path="sum"),
    ]
