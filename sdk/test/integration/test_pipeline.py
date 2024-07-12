import pytest
from bettmensch_ai import InputParameter, component, pipeline, torch_component
from bettmensch_ai.pipeline import delete, get, list
from bettmensch_ai.scripts.example_components import (
    add,
    convert_to_artifact,
    show_artifact,
    show_parameter,
    torch_ddp,
)


@pytest.mark.order(1)
def test_artifact_pipeline_decorator_and_register_and_run(
    test_output_dir, test_namespace
):
    """Defines, registers and runs a Pipeline passing artifacts across components."""

    convert_component_factory = component(convert_to_artifact)
    show_component_factory = component(show_artifact)

    @pipeline("test-artifact-pipeline", test_namespace, True)
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
    assert parameter_to_artifact_pipeline.registered_namespace == test_namespace

    parameter_to_artifact_flow = parameter_to_artifact_pipeline.run(
        {
            "a": "First integration test value a",
        },
        wait=True,
    )

    assert parameter_to_artifact_flow.status.phase == "Succeeded"


@pytest.mark.order(2)
def test_parameter_pipeline_decorator_and_register_and_run(
    test_output_dir, test_namespace
):
    """Defines, registers and runs a Pipeline passing parameters across components."""

    add_component_factory = component(add)

    @pipeline("test-parameter-pipeline", test_namespace, True)
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
    assert adding_parameters_pipeline.registered_namespace == test_namespace

    adding_parameters_flow = adding_parameters_pipeline.run(
        {"a": -100, "b": 100}, wait=True
    )

    assert adding_parameters_flow.status.phase == "Succeeded"


@pytest.mark.order(3)
def test_torch_pipeline_decorator_and_register_and_run(
    test_output_dir, test_namespace
):
    """Defines, registers and runs a Pipeline containing a non-trivial TorchComponent."""

    torch_ddp_factory = torch_component(torch_ddp)
    show_parameter_factory = component(show_parameter)

    @pipeline("test-torch-pipeline", test_namespace, True)
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
    assert torch_ddp_pipeline.registered_namespace == test_namespace

    torch_ddp_flow = torch_ddp_pipeline.run(
        {"n_iter": 12, "n_seconds_sleep": 5}, wait=True
    )

    assert torch_ddp_flow.status.phase == "Succeeded"


@pytest.mark.order(4)
@pytest.mark.parametrize(
    "test_registered_pipeline_name_pattern,test_n_registered_pipelines",
    [
        ("test-artifact-pipeline-", 1),
        ("test-parameter-pipeline-", 1),
        ("test-torch-pipeline-", 1),
        ("test-", 3),
    ],
)
def test_list(
    test_namespace,
    test_registered_pipeline_name_pattern,
    test_n_registered_pipelines,
):
    """Test the pipeline.list function."""
    registered_pipelines = list(
        registered_namespace=test_namespace,
        registered_name_pattern=test_registered_pipeline_name_pattern,
    )

    assert len(registered_pipelines) == test_n_registered_pipelines

    for registered_pipeline in registered_pipelines:
        assert registered_pipeline.registered
        assert registered_pipeline.registered_id is not None
        assert registered_pipeline.registered_name.startswith(
            f"pipeline-{test_registered_pipeline_name_pattern}"
        )
        assert registered_pipeline.registered_namespace == test_namespace


@pytest.mark.order(5)
@pytest.mark.parametrize(
    "test_registered_pipeline_name_pattern,test_pipeline_inputs",
    [
        (
            "test-artifact-pipeline-",
            {
                "a": "Second integration test value a",
            },
        ),
        ("test-parameter-pipeline-", {"a": -10, "b": 20}),
        ("test-torch-pipeline-", {"n_iter": 5, "n_seconds_sleep": 1}),
    ],
)
def test_get_and_run_from_registry(
    test_namespace, test_registered_pipeline_name_pattern, test_pipeline_inputs
):
    """Test the pipeline.get function, and the Pipeline's constructor when
    using the registered WorkflowTemplate on the Argo server as a source."""

    # we use the `list` method to retrieve the pipeline and access its
    # `registered_name` property to use as an input for the `get` method.
    registered_pipeline_name = list(
        registered_namespace=test_namespace,
        registered_name_pattern=test_registered_pipeline_name_pattern,
    )[0].registered_name

    registered_pipeline = get(registered_pipeline_name, test_namespace)

    assert registered_pipeline.registered
    assert registered_pipeline.registered_id is not None
    assert registered_pipeline.registered_name.startswith(
        f"pipeline-{test_registered_pipeline_name_pattern}"
    )
    assert registered_pipeline.registered_namespace == test_namespace

    flow = registered_pipeline.run(test_pipeline_inputs, wait=True)
    assert flow.status.phase == "Succeeded"


@pytest.mark.order(6)
def test_delete(test_namespace):
    """Test the pipeline.delete function"""

    registered_pipelines = list(
        registered_namespace=test_namespace, registered_name_pattern="test-"
    )

    for registered_pipeline in registered_pipelines:
        delete(
            registered_name=registered_pipeline.registered_name,
            registered_namespace=test_namespace,
        )

    assert (
        list(
            registered_namespace=test_namespace, registered_name_pattern="test-"
        )
        == []
    )
