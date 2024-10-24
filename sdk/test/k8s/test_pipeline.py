import pytest
from bettmensch_ai.components.examples import (
    add_parameters_factory,
    convert_to_artifact_factory,
    lightning_train_torch_ddp_factory,
    show_artifact_factory,
    show_parameter_factory,
    tensor_reduce_torch_ddp_factory,
)
from bettmensch_ai.constants import COMPONENT_IMAGE
from bettmensch_ai.io import InputParameter
from bettmensch_ai.pipelines import (
    delete_registered_pipeline,
    get_registered_pipeline,
    list_registered_pipelines,
    pipeline,
)


@pytest.mark.standard
@pytest.mark.order(1)
def test_artifact_pipeline_decorator_and_register_and_run(
    test_output_dir, test_namespace
):
    """Defines, registers and runs a Pipeline passing artifacts across
    components."""

    @pipeline("test-artifact-pipeline", test_namespace, True)
    def parameter_to_artifact_pipeline(
        a: InputParameter = "Param A",
    ) -> None:
        convert = convert_to_artifact_factory(
            "convert-to-artifact",
            a=a,
        )

        show_artifact_factory(
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
    assert (
        parameter_to_artifact_pipeline.registered_namespace == test_namespace
    )  # noqa: E501

    parameter_to_artifact_flow = parameter_to_artifact_pipeline.run(
        {
            "a": "First integration test value a",
        },
        wait=True,
    )

    assert parameter_to_artifact_flow.status.phase == "Succeeded"


@pytest.mark.standard
@pytest.mark.order(2)
def test_parameter_pipeline_decorator_and_register_and_run(
    test_output_dir, test_namespace
):
    """Defines, registers and runs a Pipeline passing parameters across
    components."""

    @pipeline("test-parameter-pipeline", test_namespace, True)
    def adding_parameters_pipeline(
        a: InputParameter = 1, b: InputParameter = 2
    ) -> None:
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


@pytest.mark.standard
@pytest.mark.order(3)
@pytest.mark.parametrize(
    "test_registered_pipeline_name_pattern,test_n_registered_pipelines",
    [
        ("test-artifact-pipeline-", 1),
        ("test-parameter-pipeline-", 1),
        ("test-", 2),
    ],
)
def test_list_registered_standard_pipelines(
    test_namespace,
    test_registered_pipeline_name_pattern,
    test_n_registered_pipelines,
):
    """Test the pipeline.list function."""
    registered_pipelines = list_registered_pipelines(
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


@pytest.mark.standard
@pytest.mark.order(4)
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
    ],
)
def test_run_standard_registered_pipelines_from_registry(
    test_namespace, test_registered_pipeline_name_pattern, test_pipeline_inputs
):
    """Test the pipeline.get function, and the Pipeline's constructor when
    using the registered WorkflowTemplate on the Argo server as a source."""

    # we use the `list` method to retrieve the pipeline and access its
    # `registered_name` property to use as an input for the `get` method.
    registered_pipeline_name = list_registered_pipelines(
        registered_namespace=test_namespace,
        registered_name_pattern=test_registered_pipeline_name_pattern,
    )[0].registered_name

    registered_pipeline = get_registered_pipeline(
        registered_pipeline_name, test_namespace
    )

    assert registered_pipeline.registered
    assert registered_pipeline.registered_id is not None
    assert registered_pipeline.registered_name.startswith(
        f"pipeline-{test_registered_pipeline_name_pattern}"
    )
    assert registered_pipeline.registered_namespace == test_namespace

    flow = registered_pipeline.run(test_pipeline_inputs, wait=True)
    assert flow.status.phase == "Succeeded"


@pytest.mark.ddp
@pytest.mark.order(5)
@pytest.mark.parametrize(
    "test_pipeline_name, test_n_nodes, test_gpus, test_memory",
    [
        ("test-torch-cpu-pipeline", 2, None, "300Mi"),
        ("test-torch-gpu-pipeline", 2, 1, "700Mi"),
    ],
)
def test_torch_ddp_pipeline_decorator_and_register_and_run(
    test_pipeline_name,
    test_n_nodes,
    test_gpus,
    test_memory,
    test_output_dir,
    test_namespace,
):
    """Defines, registers and runs a Pipeline containing a non-trivial
    TorchComponent. The pod spec patch ensures distributing replica pods of the
    TorchComponent across different K8s nodes."""

    @pipeline(test_pipeline_name, test_namespace, True)
    def torch_ddp_pipeline(
        n_iter: InputParameter, n_seconds_sleep: InputParameter
    ) -> None:
        torch_ddp_test = (
            tensor_reduce_torch_ddp_factory(
                "torch-ddp",
                hera_template_kwargs={
                    "pod_spec_patch": """topologySpreadConstraints:
- maxSkew: 1
  topologyKey: kubernetes.io/hostname
  whenUnsatisfiable: DoNotSchedule
  labelSelector:
    matchExpressions:
      - { key: torch-node, operator: In, values: ['0','1','2','3','4','5']}"""
                },
                n_nodes=test_n_nodes,
                n_iter=n_iter,
                n_seconds_sleep=n_seconds_sleep,
            )
            .set_gpus(test_gpus)
            .set_memory(test_memory)
        )

        show_parameter_factory(
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
        {"n_iter": 5, "n_seconds_sleep": 2}, wait=True
    )

    assert torch_ddp_flow.status.phase == "Succeeded"


@pytest.mark.ddp
@pytest.mark.order(6)
@pytest.mark.parametrize(
    "test_pipeline_name, test_n_nodes, test_gpus, test_memory",
    [
        ("test-lightning-cpu-pipeline", 2, None, "1Gi"),
        ("test-lightning-gpu-pipeline", 2, 1, "1Gi"),
    ],
)
def test_lightning_ddp_pipeline_decorator_and_register_and_run(
    test_pipeline_name,
    test_n_nodes,
    test_gpus,
    test_memory,
    test_output_dir,
    test_namespace,
):
    """Defines, registers and runs a Pipeline containing a non-trivial
    TorchComponent. The pod spec patch ensures distributing replica pods of the
    TorchComponent across different K8s nodes."""

    @pipeline(test_pipeline_name, test_namespace, True)
    def lightning_ddp_pipeline(
        max_time: InputParameter,
    ) -> None:
        lightning_ddp_test = (
            lightning_train_torch_ddp_factory(
                "lightning-ddp",
                hera_template_kwargs={
                    "image": COMPONENT_IMAGE.lightning.value,
                    "pod_spec_patch": """topologySpreadConstraints:
- maxSkew: 1
  topologyKey: kubernetes.io/hostname
  whenUnsatisfiable: DoNotSchedule
  labelSelector:
    matchExpressions:
      - { key: torch-node, operator: In, values: ['0','1','2','3','4','5']}""",
                },
                n_nodes=test_n_nodes,
                max_time=max_time,
            )
            .set_gpus(test_gpus)
            .set_memory(test_memory)
        )

        show_parameter_factory(
            "show-duration-param",
            a=lightning_ddp_test.outputs["duration"],
        )

    lightning_ddp_pipeline.export(test_output_dir)

    assert not lightning_ddp_pipeline.registered
    assert lightning_ddp_pipeline.registered_id is None
    assert lightning_ddp_pipeline.registered_name is None
    assert lightning_ddp_pipeline.registered_namespace is None

    lightning_ddp_pipeline.register()

    assert lightning_ddp_pipeline.registered
    assert lightning_ddp_pipeline.registered_id is not None
    assert lightning_ddp_pipeline.registered_name.startswith(
        f"pipeline-{lightning_ddp_pipeline.name}-"
    )
    assert lightning_ddp_pipeline.registered_namespace == test_namespace

    lightning_ddp_flow = lightning_ddp_pipeline.run(
        {"max_time": "00:00:00:20"}, wait=True
    )

    assert lightning_ddp_flow.status.phase == "Succeeded"


@pytest.mark.ddp
@pytest.mark.order(7)
@pytest.mark.parametrize(
    "test_registered_pipeline_name_pattern,test_n_registered_pipelines",
    [
        ("test-torch-cpu-pipeline-", 1),
        ("test-torch-gpu-pipeline-", 1),
        ("test-lightning-cpu-pipeline-", 1),
        ("test-lightning-gpu-pipeline-", 1),
    ],
)
def test_list_registered_ddp_pipelines(
    test_namespace,
    test_registered_pipeline_name_pattern,
    test_n_registered_pipelines,
):
    """Test the pipeline.list function."""
    registered_pipelines = list_registered_pipelines(
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


@pytest.mark.ddp
@pytest.mark.order(8)
@pytest.mark.parametrize(
    "test_registered_pipeline_name_pattern,test_pipeline_inputs",
    [
        ("test-torch-gpu-pipeline-", {"n_iter": 5, "n_seconds_sleep": 2}),
        ("test-torch-cpu-pipeline-", {"n_iter": 5, "n_seconds_sleep": 2}),
        (
            "test-lightning-gpu-pipeline-",
            {
                "max_time": "00:00:00:30",
            },
        ),
        (
            "test-lightning-cpu-pipeline-",
            {
                "max_time": "00:00:00:30",
            },
        ),
    ],
)
def test_run_dpp_registered_pipelines_from_registry(
    test_namespace, test_registered_pipeline_name_pattern, test_pipeline_inputs
):
    """Test the pipeline.get function, and the Pipeline's constructor when
    using the registered WorkflowTemplate on the Argo server as a source."""

    # we use the `list` method to retrieve the pipeline and access its
    # `registered_name` property to use as an input for the `get` method.
    registered_pipeline_name = list_registered_pipelines(
        registered_namespace=test_namespace,
        registered_name_pattern=test_registered_pipeline_name_pattern,
    )[0].registered_name

    registered_pipeline = get_registered_pipeline(
        registered_pipeline_name, test_namespace
    )

    assert registered_pipeline.registered
    assert registered_pipeline.registered_id is not None
    assert registered_pipeline.registered_name.startswith(
        f"pipeline-{test_registered_pipeline_name_pattern}"
    )
    assert registered_pipeline.registered_namespace == test_namespace

    flow = registered_pipeline.run(test_pipeline_inputs, wait=True)
    assert flow.status.phase == "Succeeded"


@pytest.mark.standard
@pytest.mark.ddp
@pytest.mark.delete_pipelines
@pytest.mark.order(9)
def test_delete(test_namespace):
    """Test the pipeline.delete function"""

    registered_pipelines = list_registered_pipelines(
        registered_namespace=test_namespace, registered_name_pattern="test-"
    )

    for registered_pipeline in registered_pipelines:
        delete_registered_pipeline(
            registered_name=registered_pipeline.registered_name,
            registered_namespace=test_namespace,
        )

    assert (
        list_registered_pipelines(
            registered_namespace=test_namespace,
            registered_name_pattern="test-",  # noqa: E501
        )
        == []
    )
