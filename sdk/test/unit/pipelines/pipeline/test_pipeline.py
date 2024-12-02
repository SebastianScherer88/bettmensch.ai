from bettmensch_ai.pipelines.io import InputParameter
from bettmensch_ai.pipelines.pipeline.examples import (
    adding_parameters_pipeline,
    parameter_to_artifact_pipeline,
)
from hera.workflows import Artifact, Parameter, models


def test_artifact_pipeline(
    test_output_dir,
):
    """Declaration of Pipeline using InputArtifact and OutputArtifact"""

    assert parameter_to_artifact_pipeline.built
    assert not parameter_to_artifact_pipeline.registered
    assert parameter_to_artifact_pipeline.registered_id is None
    assert parameter_to_artifact_pipeline.registered_name is None
    assert parameter_to_artifact_pipeline.registered_namespace is None

    # --- validate class' io attributes
    assert parameter_to_artifact_pipeline.inputs == {
        "a": InputParameter(name="a", value="Param A").set_owner(
            parameter_to_artifact_pipeline
        ),
    }
    assert parameter_to_artifact_pipeline.required_inputs == {}
    # we dont have access to the output's Component type owner, so can only
    # verify the name here.
    assert list(parameter_to_artifact_pipeline.outputs.keys()) == [
        "b",
    ]
    assert parameter_to_artifact_pipeline.task_inputs == {
        "a": InputParameter(name="a")
        .set_owner(parameter_to_artifact_pipeline)
        .set_source(parameter_to_artifact_pipeline.inputs["a"]),
    }

    # --- validate class' `user_built_workflow_template` attribute
    wft = parameter_to_artifact_pipeline.user_built_workflow_template

    # validate spec
    assert wft.entrypoint == "bettmensch-ai-outer-dag"
    assert wft.arguments == [Parameter(name="a", value="Param A")]

    # validate inner dag template
    inner_dag_template = wft.templates[0]
    assert inner_dag_template.name == "bettmensch-ai-inner-dag"
    assert inner_dag_template.inputs == [
        Parameter(name="a", value="Param A"),
    ]
    assert inner_dag_template.outputs == [
        Artifact(
            name="b", from_="{{tasks.show-artifact-0.outputs.artifacts.b}}"
        ),
    ]
    inner_dag_template_tasks = inner_dag_template.tasks
    assert [task.name for task in inner_dag_template_tasks] == [
        "convert-to-artifact-0",
        "show-artifact-0",
    ]

    # validate script templates
    script_template_names = [template.name for template in wft.templates[1:-1]]
    assert script_template_names == [
        "convert-to-artifact",
        "show-artifact",
    ]

    # validate outer dag template
    outer_dag_template = wft.templates[-1]
    assert outer_dag_template.name == "bettmensch-ai-outer-dag"
    outer_dag_template_task = outer_dag_template.tasks[0]
    assert outer_dag_template_task.name == "bettmensch-ai-inner-dag"
    assert outer_dag_template_task.arguments == [
        Parameter(name="a", value="{{workflow.parameters.a}}"),
    ]

    parameter_to_artifact_pipeline.export(test_output_dir)


def test_parameter_pipeline(test_output_dir):
    """Declaration of Pipeline using InputParameter and OutputParameter"""

    # --- validate class' basic attributes
    assert adding_parameters_pipeline.built
    assert not adding_parameters_pipeline.registered
    assert adding_parameters_pipeline.registered_id is None
    assert adding_parameters_pipeline.registered_name is None
    assert adding_parameters_pipeline.registered_namespace is None

    # --- validate class' io attributes
    assert adding_parameters_pipeline.inputs == {
        "a": InputParameter(name="a", value=1).set_owner(
            adding_parameters_pipeline
        ),
        "b": InputParameter(name="b", value=2).set_owner(
            adding_parameters_pipeline
        ),
    }
    assert adding_parameters_pipeline.required_inputs == {}
    # we dont have access to the output's Component type owner, so can only
    # verify the name here.
    assert list(adding_parameters_pipeline.outputs.keys()) == [
        "sum",
    ]
    assert adding_parameters_pipeline.task_inputs == {
        "a": InputParameter(name="a")
        .set_owner(adding_parameters_pipeline)
        .set_source(adding_parameters_pipeline.inputs["a"]),
        "b": InputParameter(name="b")
        .set_owner(adding_parameters_pipeline)
        .set_source(adding_parameters_pipeline.inputs["b"]),
    }

    # --- validate class' `user_built_workflow_template` attribute
    wft = adding_parameters_pipeline.user_built_workflow_template

    # validate spec
    assert wft.entrypoint == "bettmensch-ai-outer-dag"
    assert wft.arguments == [
        Parameter(name="a", value=1),
        Parameter(name="b", value=2),
    ]

    # validate inner dag template
    inner_dag_template = wft.templates[0]
    assert inner_dag_template.name == "bettmensch-ai-inner-dag"
    assert inner_dag_template.inputs == [
        Parameter(name="a", value=1),
        Parameter(name="b", value=2),
    ]
    assert inner_dag_template.outputs == [
        Parameter(
            name="sum",
            value_from=models.ValueFrom(
                parameter="{{tasks.a-plus-b-plus-2-0.outputs.parameters.sum}}"
            ),
        ),
    ]
    inner_dag_template_tasks = inner_dag_template.tasks
    assert [task.name for task in inner_dag_template_tasks] == [
        "a-plus-b-0",
        "a-plus-b-plus-2-0",
    ]

    # validate script templates
    script_template_names = [template.name for template in wft.templates[1:-1]]
    assert script_template_names == [
        "a-plus-b",
        "a-plus-b-plus-2",
    ]

    # validate outer dag template
    outer_dag_template = wft.templates[-1]
    assert outer_dag_template.name == "bettmensch-ai-outer-dag"
    outer_dag_template_task = outer_dag_template.tasks[0]
    assert outer_dag_template_task.name == "bettmensch-ai-inner-dag"
    assert outer_dag_template_task.arguments == [
        Parameter(name="a", value="{{workflow.parameters.a}}"),
        Parameter(name="b", value="{{workflow.parameters.b}}"),
    ]

    adding_parameters_pipeline.export(test_output_dir)
