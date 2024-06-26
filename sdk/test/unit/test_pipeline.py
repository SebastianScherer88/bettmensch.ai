from bettmensch_ai import (
    PIPELINE_TYPE,
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
    component,
    pipeline,
)
from hera.workflows import WorkflowTemplate


def test_artifact_pipeline(
    test_convert_to_artifact_function, test_show_artifact_function
):
    """Declaration of Pipeline using InputArtifact and OutputArtifact"""

    convert_component_factory = component(test_convert_to_artifact_function)
    show_component_factory = component(test_show_artifact_function)

    @pipeline("test-artifact-pipeline", "argo", True)
    def parameter_to_artifact(
        a: InputParameter = "Param A",
        b: InputParameter = "Param B",
    ) -> None:
        convert = convert_component_factory(
            "convert-to-artifact",
            a=a,
            b=b,
        )

        show = show_component_factory(
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

    task_names = [task.name for task in wft.templates[0].tasks]
    assert task_names == ["convert-to-artifact-0", "show-artifact-0"]

    script_template_names = [template.name for template in wft.templates[1:]]
    assert script_template_names == ["convert-to-artifact", "show-artifact"]


def test_parameter_pipeline(test_add_function):
    """Declaration of Pipeline using InputParameter and OutputParameter"""

    add_component_factory = component(test_add_function)

    @pipeline("test-parameter-pipeline", "argo", True)
    def adding_parameters(a: InputParameter = 1, b: InputParameter = 2) -> None:
        a_plus_b = add_component_factory(
            "a-plus-b",
            a=a,
            b=b,
        )

        a_plus_b_plus_2 = add_component_factory(
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
    assert task_names == ["a-plus-b-0", "a-plus-b-plus-2-0"]

    script_template_names = [template.name for template in wft.templates[1:]]
    assert script_template_names == ["a-plus-b", "a-plus-b-plus-2"]
