from bettmensch_ai import InputParameter, component, pipeline, torch_component
from bettmensch_ai.pipeline import delete, get, list
from bettmensch_ai.scripts.example_components import (
    add,
    convert_to_artifact,
    show_artifact,
    show_parameter,
    torch_ddp,
)


def test_artifact_pipeline_decorator_and_register_and_run(
    test_output_dir,
):
    """Defines, registers and runs a Pipeline passing artifacts across components."""

    convert_component_factory = component(convert_to_artifact)
    show_component_factory = component(show_artifact)

    @pipeline("test-artifact-pipeline", "argo", True)
    def parameter_to_artifact_pipeline(
        a: InputParameter = "Param A",
    ) -> None:
        convert = convert_component_factory(
            "convert-to-artifact",
            a=a,
        )

        show = show_component_factory(
            "show-artifact",
            a=convert.outputs["a_art"],
        )

    parameter_to_artifact_pipeline.export(test_output_dir)

    assert not parameter_to_artifact_pipeline.registered
    assert parameter_to_artifact_pipeline.registered_id is None
    assert parameter_to_artifact_pipeline.registered_name is None
    assert parameter_to_artifact_pipeline.registered_namespace is None

    parameter_to_artifact_pipeline.register()

    assert parameter_to_artifact_pipeline.registered
    assert parameter_to_artifact_pipeline.registered_id is not None
    assert parameter_to_artifact_pipeline.registered_name.startswith(
        f"pipeline-{parameter_to_artifact_pipeline.name}-"
    )
    assert parameter_to_artifact_pipeline.registered_namespace == "argo"

    parameter_to_artifact_flow = parameter_to_artifact_pipeline.run(
        {
            "a": "Integration test value a",
        },
        wait=True,
    )

    assert parameter_to_artifact_flow.status.phase == "Succeeded"


def test_parameter_pipeline_decorator_and_register_and_run(test_output_dir):
    """Defines, registers and runs a Pipeline passing parameters across components."""

    add_component_factory = component(add)

    @pipeline("test-parameter-pipeline", "argo", True)
    def adding_parameters_pipeline(
        a: InputParameter = 1, b: InputParameter = 2
    ) -> None:
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

    adding_parameters_pipeline.export(test_output_dir)

    assert not adding_parameters_pipeline.registered
    assert adding_parameters_pipeline.registered_id is None
    assert adding_parameters_pipeline.registered_name is None
    assert adding_parameters_pipeline.registered_namespace is None

    adding_parameters_pipeline.register()

    assert adding_parameters_pipeline.registered
    assert adding_parameters_pipeline.registered_id is not None
    assert adding_parameters_pipeline.registered_name.startswith(
        f"pipeline-{adding_parameters_pipeline.name}-"
    )
    assert adding_parameters_pipeline.registered_namespace == "argo"

    adding_parameters_flow = adding_parameters_pipeline.run(
        {"a": -100, "b": 100}, wait=True
    )

    assert adding_parameters_flow.status.phase == "Succeeded"


def test_torch_pipeline_decorator_and_register_and_run(test_output_dir):
    """Defines, registers and runs a Pipeline containing a non-trivial TorchComponent."""

    torch_ddp_factory = torch_component(torch_ddp)
    show_parameter_factory = component(show_parameter)

    @pipeline("test-torch-pipeline", "argo", True)
    def torch_ddp_pipeline(
        n_iter: InputParameter, n_seconds_sleep: InputParameter
    ) -> None:
        torch_ddp_test = torch_ddp_factory(
            "torch-ddp",
            n_nodes=3,
            n_iter=n_iter,
            n_seconds_sleep=n_seconds_sleep,
        )

        torch_ddp_duration_1 = show_parameter_factory(
            "show-duration-param",
            a=torch_ddp_test.outputs["duration"],
        )

    torch_ddp_pipeline.export(test_output_dir)

    assert not torch_ddp_pipeline.registered
    assert torch_ddp_pipeline.registered_id is None
    assert torch_ddp_pipeline.registered_name is None
    assert torch_ddp_pipeline.registered_namespace is None

    torch_ddp_pipeline.register()

    assert torch_ddp_pipeline.registered
    assert torch_ddp_pipeline.registered_id is not None
    assert torch_ddp_pipeline.registered_name.startswith(
        f"pipeline-{torch_ddp_pipeline.name}-"
    )
    assert torch_ddp_pipeline.registered_namespace == "argo"

    torch_ddp_flow = torch_ddp_pipeline.run(
        {"n_iter": 12, "n_seconds_sleep": 5}, wait=True
    )

    assert torch_ddp_flow.status.phase == "Succeeded"


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
