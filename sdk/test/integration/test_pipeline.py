from bettmensch_ai import (
    InputParameter,
    OutputParameter,
    Pipeline,
    component,
    pipeline,
    torch_component,
)
from bettmensch_ai.pipeline import delete, get, list


def test_artifact_pipeline_decorator_and_register_and_run(
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

    assert not parameter_to_artifact.registered
    assert parameter_to_artifact.registered_id is None
    assert parameter_to_artifact.registered_name is None
    assert parameter_to_artifact.registered_namespace is None

    parameter_to_artifact.register()

    assert parameter_to_artifact.registered
    assert parameter_to_artifact.registered_id is not None
    assert parameter_to_artifact.registered_name.startswith(
        f"pipeline-{parameter_to_artifact.name}-"
    )
    assert parameter_to_artifact.registered_namespace == "argo"

    parameter_to_artifact.run(
        {"a": "Integration test value a", "b": "Integration test value b"}
    )


def test_parameter_pipeline_decorator_and_register_and_run(test_add_function):
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

    assert not adding_parameters.registered
    assert adding_parameters.registered_id is None
    assert adding_parameters.registered_name is None
    assert adding_parameters.registered_namespace is None

    adding_parameters.register()

    assert adding_parameters.registered
    assert adding_parameters.registered_id is not None
    assert adding_parameters.registered_name.startswith(
        f"pipeline-{adding_parameters.name}-"
    )
    assert adding_parameters.registered_namespace == "argo"

    adding_parameters.run({"a": -100, "b": 100})


def test_torch_pipeline_decorator_and_register_and_run():
    """Declaration of Pipeline using InputParameter and OutputParameter"""

    @torch_component
    def test_torch_ddp_function(
        n_iter: InputParameter,
        n_seconds_sleep: InputParameter,
        duration: OutputParameter,
    ):
        from bettmensch_ai.scripts import torch_ddp_test

        torch_ddp_test(n_iter, n_seconds_sleep)

        duration.assign(n_iter * n_seconds_sleep)

    @component
    def show_parameter(torch_duration: InputParameter):

        print(
            f"The previous torch ddp component took {torch_duration} seconds."
        )

    @pipeline("test-torch-pipeline", "argo", True)
    def torch_ddp(
        n_iter: InputParameter, n_seconds_sleep: InputParameter
    ) -> None:
        torch_ddp_test = test_torch_ddp_function(
            "torch-ddp",
            n_nodes=2,
            n_iter=n_iter,
            n_seconds_sleep=n_seconds_sleep,
        )

        torch_ddp_duration = show_parameter(
            "show-torch-ddp-duration",
            torch_duration=torch_ddp_test.outputs["duration"],
        )

    assert not torch_ddp.registered
    assert torch_ddp.registered_id is None
    assert torch_ddp.registered_name is None
    assert torch_ddp.registered_namespace is None

    torch_ddp.register()

    assert torch_ddp.registered
    assert torch_ddp.registered_id is not None
    assert torch_ddp.registered_name.startswith(f"pipeline-{torch_ddp.name}-")
    assert torch_ddp.registered_namespace == "argo"

    torch_ddp.run({"n_iter": 12, "n_seconds_sleep": 5})


def test_list():
    """Test the pipeline.list function."""
    registered_pipelines = list(
        registered_namespace="argo", registered_name_pattern="test-"
    )

    for registered_pipeline in registered_pipelines:
        assert registered_pipeline.registered
        assert registered_pipeline.registered_id is not None
        assert registered_pipeline.registered_name.startswith(
            f"pipeline-{registered_pipeline.name}-"
        )
        assert registered_pipeline.registered_namespace == "argo"


# def test_get():
#     """Test the pipeline.get function, implicitly using the
#     Pipeline.from_workflow_template method."""

#     registered_pipelines = list(
#         registered_namespace='argo',
#         registered_name_pattern='test-')

#     parameters_to_artifact_pipeline = get(registered_pipelines[0].registered_name,registered_pipelines[0].registered_namespace)
#     adding_parameters_pipeline = get(registered_pipelines[1].registered_name,registered_pipelines[1].registered_namespace)

#     # import pdb
#     # pdb.set_trace()

#     parameters_to_artifact_pipeline.run({'a':'Test value a','b':'Test value b'})
#     adding_parameters_pipeline.run({'a':-100,'b':100})

# def test_pipeline_run():
#     """Test the Pipeline.run method."""

# def test_delete():
#     """Test the pipeline.delete function"""

#     registered_pipelines = list(
#         registered_namespace='argo',
#         registered_name_pattern='test-')

#     for registered_pipeline in registered_pipelines:
#         delete(registered_name=registered_pipeline.registered_name,registered_namespace=registered_pipeline.registered_namespace)

#     assert list(
#         registered_namespace='argo',
#         registered_name_pattern='test-') == []
