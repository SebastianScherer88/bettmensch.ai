from bettmensch_ai.components import component, torch_component
from bettmensch_ai.io import InputParameter
from bettmensch_ai.pipelines import pipeline
from hera.workflows import WorkflowTemplate


def test_artifact_pipeline(
    test_convert_to_artifact_function,
    test_show_artifact_function,
    test_output_dir,
):
    """Declaration of Pipeline using InputArtifact and OutputArtifact"""

    convert_torch_component_factory = torch_component(
        test_convert_to_artifact_function
    )
    show_component_factory = component(test_show_artifact_function)

    @pipeline("test-artifact-pipeline", "argo", True)
    def parameter_to_artifact(
        a: InputParameter = "Param A",
        b: InputParameter = "Param B",
    ) -> None:
        convert = convert_torch_component_factory(
            "convert-to-artifact",
            n_nodes=2,
            a=a,
            b=b,
        )

        show_component_factory(
            "show-artifact",
            a=convert.outputs["a_art"],
            b=convert.outputs["b_art"],
        )

    assert parameter_to_artifact.built
    assert not parameter_to_artifact.registered
    assert parameter_to_artifact.registered_id is None
    assert parameter_to_artifact.registered_name is None
    assert parameter_to_artifact.registered_namespace is None
    assert set(parameter_to_artifact.inputs.keys()) == {"a", "b"}
    for pipeline_input_name, pipeline_input_default in (
        ("a", "Param A"),
        ("b", "Param B"),
    ):
        assert (
            parameter_to_artifact.inputs[pipeline_input_name].name
            == pipeline_input_name
        )
        assert (
            parameter_to_artifact.inputs[pipeline_input_name].owner
            == parameter_to_artifact
        )
        assert (
            parameter_to_artifact.inputs[pipeline_input_name].value
            == pipeline_input_default
        )
    assert isinstance(
        parameter_to_artifact.user_built_workflow_template, WorkflowTemplate
    )

    wft = parameter_to_artifact.user_built_workflow_template

    task_names = [task.name for task in wft.templates[2].tasks]
    assert task_names == [
        "convert-to-artifact-create-torch-service",
        "convert-to-artifact-0",
        "convert-to-artifact-0-worker-1",
        "convert-to-artifact-delete-torch-service",
        "show-artifact-0",
    ]

    script_template_names = [template.name for template in wft.templates]
    assert script_template_names == [
        "convert-to-artifact-create-torch-service",
        "convert-to-artifact-delete-torch-service",
        "bettmensch-ai-dag",
        "convert-to-artifact-0",
        "convert-to-artifact-1",
        "show-artifact",
    ]

    assert wft.templates[3].labels["torch-node"] == "0"
    assert (
        wft.templates[3]
        .labels["torch-job"]
        .startswith("convert-to-artifact-0-")
    )
    assert wft.templates[4].labels["torch-node"] == "1"

    parameter_to_artifact.export(test_output_dir)


def test_parameter_pipeline(test_add_function, test_output_dir):
    """Declaration of Pipeline using InputParameter and OutputParameter"""

    add_component_factory = component(test_add_function)
    add_torch_component_factory = torch_component(test_add_function)

    @pipeline("test-parameter-pipeline", "argo", True)
    def adding_parameters(
        a: InputParameter = 1, b: InputParameter = 2
    ) -> None:  # noqa: E501
        a_plus_b = add_torch_component_factory(
            "a-plus-b",
            n_nodes=2,
            a=a,
            b=b,
        )

        add_component_factory(
            "a-plus-b-plus-2",
            a=a_plus_b.outputs["sum"],
            b=InputParameter("two", 2),
        )

    assert adding_parameters.built
    assert not adding_parameters.registered
    assert adding_parameters.registered_id is None
    assert adding_parameters.registered_name is None
    assert adding_parameters.registered_namespace is None
    assert set(adding_parameters.inputs.keys()) == {"a", "b"}
    for pipeline_input_name, pipeline_input_default in (("a", 1), ("b", 2)):
        assert (
            adding_parameters.inputs[pipeline_input_name].name
            == pipeline_input_name
        )
        assert (
            adding_parameters.inputs[pipeline_input_name].owner
            == adding_parameters
        )
        assert (
            adding_parameters.inputs[pipeline_input_name].value
            == pipeline_input_default
        )

    wft = adding_parameters.user_built_workflow_template

    task_names = [task.name for task in wft.templates[2].tasks]
    assert task_names == [
        "a-plus-b-create-torch-service",
        "a-plus-b-0",
        "a-plus-b-0-worker-1",
        "a-plus-b-delete-torch-service",
        "a-plus-b-plus-2-0",
    ]

    script_template_names = [template.name for template in wft.templates]
    assert script_template_names == [
        "a-plus-b-create-torch-service",
        "a-plus-b-delete-torch-service",
        "bettmensch-ai-dag",
        "a-plus-b-0",
        "a-plus-b-1",
        "a-plus-b-plus-2",
    ]

    assert wft.templates[3].labels["torch-node"] == "0"
    assert wft.templates[3].labels["torch-job"].startswith("a-plus-b-0-")
    assert wft.templates[4].labels["torch-node"] == "1"

    adding_parameters.export(test_output_dir)
