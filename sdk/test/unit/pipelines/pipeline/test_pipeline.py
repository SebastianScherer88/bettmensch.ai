from bettmensch_ai.pipelines.pipeline.examples import (
    adding_parameters_pipeline,
    parameter_to_artifact_pipeline,
)
from hera.workflows import WorkflowTemplate


def test_artifact_pipeline(
    test_output_dir,
):
    """Declaration of Pipeline using InputArtifact and OutputArtifact"""

    assert parameter_to_artifact_pipeline.built
    assert not parameter_to_artifact_pipeline.registered
    assert parameter_to_artifact_pipeline.registered_id is None
    assert parameter_to_artifact_pipeline.registered_name is None
    assert parameter_to_artifact_pipeline.registered_namespace is None
    assert set(
        parameter_to_artifact_pipeline.inner_dag_template_inputs.keys()
    ) == {"a"}
    assert set(
        parameter_to_artifact_pipeline.inner_dag_template_outputs.keys()
    ) == {"b"}
    assert set(
        parameter_to_artifact_pipeline.inner_dag_task_inputs.keys()
    ) == {"a"}
    for pipeline_input_name, pipeline_input_default in (("a", "Param A"),):
        assert (
            parameter_to_artifact_pipeline.inner_dag_template_inputs[
                pipeline_input_name
            ].name
            == pipeline_input_name
        )
        assert (
            parameter_to_artifact_pipeline.inner_dag_template_inputs[
                pipeline_input_name
            ].owner
            == parameter_to_artifact_pipeline
        )
        assert (
            parameter_to_artifact_pipeline.inner_dag_template_inputs[
                pipeline_input_name
            ].value
            == pipeline_input_default
        )
    assert isinstance(
        parameter_to_artifact_pipeline.user_built_workflow_template,
        WorkflowTemplate,
    )

    wft = parameter_to_artifact_pipeline.user_built_workflow_template

    task_names = [task.name for task in wft.templates[0].tasks]
    assert task_names == [
        "convert-to-artifact-0",
        "show-artifact-0",
    ]

    script_template_names = [template.name for template in wft.templates]
    assert script_template_names == [
        "bettmensch-ai-inner-dag",
        "convert-to-artifact",
        "show-artifact",
        "bettmensch-ai-outer-dag",
    ]

    parameter_to_artifact_pipeline.export(test_output_dir)


def test_parameter_pipeline(test_output_dir):
    """Declaration of Pipeline using InputParameter and OutputParameter"""

    # @as_pipeline("test-parameter-pipeline", "argo", True)
    # def adding_parameters(
    #     a: InputParameter = 1, b: InputParameter = 2
    # ) -> None:  # noqa: E501
    #     a_plus_b = add_parameters_factory(
    #         "a-plus-b",
    #         a=a,
    #         b=b,
    #     )

    #     add_parameters_factory(
    #         "a-plus-b-plus-2",
    #         a=a_plus_b.outputs["sum"],
    #         b=InputParameter("two", 2),
    #     )

    assert adding_parameters_pipeline.built
    assert not adding_parameters_pipeline.registered
    assert adding_parameters_pipeline.registered_id is None
    assert adding_parameters_pipeline.registered_name is None
    assert adding_parameters_pipeline.registered_namespace is None
    assert set(
        adding_parameters_pipeline.inner_dag_template_inputs.keys()
    ) == {"a", "b"}
    assert set(
        adding_parameters_pipeline.inner_dag_template_outputs.keys()
    ) == {"sum"}
    assert set(adding_parameters_pipeline.inner_dag_task_inputs.keys()) == {
        "a",
        "b",
    }
    for pipeline_input_name, pipeline_input_default in (("a", 1), ("b", 2)):
        assert (
            adding_parameters_pipeline.inner_dag_template_inputs[
                pipeline_input_name
            ].name
            == pipeline_input_name
        )
        assert (
            adding_parameters_pipeline.inner_dag_template_inputs[
                pipeline_input_name
            ].owner
            == adding_parameters_pipeline
        )
        assert (
            adding_parameters_pipeline.inner_dag_template_inputs[
                pipeline_input_name
            ].value
            == pipeline_input_default
        )

    wft = adding_parameters_pipeline.user_built_workflow_template

    task_names = [task.name for task in wft.templates[0].tasks]
    assert task_names == [
        "a-plus-b-0",
        "a-plus-b-plus-2-0",
    ]

    script_template_names = [template.name for template in wft.templates]
    assert script_template_names == [
        "bettmensch-ai-inner-dag",
        "a-plus-b",
        "a-plus-b-plus-2",
        "bettmensch-ai-outer-dag",
    ]

    adding_parameters_pipeline.export(test_output_dir)
