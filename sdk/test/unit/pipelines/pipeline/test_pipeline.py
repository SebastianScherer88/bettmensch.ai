from bettmensch_ai.pipelines.component.examples import (
    add_parameters_factory,
    convert_to_artifact_factory,
    show_parameter_factory,
)
from bettmensch_ai.pipelines.io import InputParameter
from bettmensch_ai.pipelines.pipeline import pipeline
from hera.workflows import WorkflowTemplate


def test_artifact_pipeline(
    test_output_dir,
):
    """Declaration of Pipeline using InputArtifact and OutputArtifact"""

    @pipeline("test-artifact-pipeline", "argo", True)
    def parameter_to_artifact(
        a: InputParameter = "Param A",
    ) -> None:
        convert = convert_to_artifact_factory(
            "convert-to-artifact",
            a=a,
        )

        show_parameter_factory(
            "show-artifact",
            a=convert.outputs["a_art"],
        )

    assert parameter_to_artifact.built
    assert not parameter_to_artifact.registered
    assert parameter_to_artifact.registered_id is None
    assert parameter_to_artifact.registered_name is None
    assert parameter_to_artifact.registered_namespace is None
    assert set(parameter_to_artifact.inputs.keys()) == {"a"}
    for pipeline_input_name, pipeline_input_default in (("a", "Param A"),):
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

    task_names = [task.name for task in wft.templates[0].tasks]
    assert task_names == [
        "convert-to-artifact-0",
        "show-artifact-0",
    ]

    script_template_names = [template.name for template in wft.templates]
    assert script_template_names == [
        "bettmensch-ai-dag",
        "convert-to-artifact",
        "show-artifact",
    ]

    parameter_to_artifact.export(test_output_dir)


def test_parameter_pipeline(test_output_dir):
    """Declaration of Pipeline using InputParameter and OutputParameter"""

    @pipeline("test-parameter-pipeline", "argo", True)
    def adding_parameters(
        a: InputParameter = 1, b: InputParameter = 2
    ) -> None:  # noqa: E501
        a_plus_b = add_parameters_factory(
            "a-plus-b",
            a=a,
            b=b,
        )

        add_parameters_factory(
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

    task_names = [task.name for task in wft.templates[0].tasks]
    assert task_names == [
        "a-plus-b-0",
        "a-plus-b-plus-2-0",
    ]

    script_template_names = [template.name for template in wft.templates]
    assert script_template_names == [
        "bettmensch-ai-dag",
        "a-plus-b",
        "a-plus-b-plus-2",
    ]

    adding_parameters.export(test_output_dir)
