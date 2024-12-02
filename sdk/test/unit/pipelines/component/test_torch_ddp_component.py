from bettmensch_ai.pipelines.component import (
    TorchDDPComponent,
    as_torch_ddp_component,
)
from bettmensch_ai.pipelines.component.examples import (
    add_parameters_torch_ddp_factory,
    convert_to_artifact_torch_ddp_factory,
    show_parameter_torch_ddp_factory,
)
from bettmensch_ai.pipelines.io import InputParameter
from bettmensch_ai.pipelines.pipeline_context import _pipeline_context
from hera.workflows import DAG, Parameter, WorkflowTemplate


def test_torch_component___init__(test_function_and_task_inputs):
    """Tests of Component constructor."""

    test_function, test_task_inputs = test_function_and_task_inputs

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        test_component = (
            TorchDDPComponent(
                func=test_function,
                name="test_name",
                n_nodes=2,
                min_nodes=2,
                nproc_per_node=2,
                **test_task_inputs
            )
            .set_cpu(0.5)
            .set_memory("100Mi")
            .set_gpus(1)
        )

    # validate addition of component to pipeline context
    assert test_component == _pipeline_context.components[0]

    # validate component attributes
    assert isinstance(test_component, TorchDDPComponent)
    assert test_component.implementation == "torch-ddp"
    assert test_component.base_name == "test-name"
    assert test_component.name == "test-name-0"
    assert test_component.func == test_function
    assert test_component.hera_template_kwargs == {}
    assert test_component.n_nodes == 2
    assert test_component.min_nodes == 2
    assert test_component.nproc_per_node == 2
    assert test_component.depends == "mock-component-0"
    assert test_component.cpu == 0.5
    assert test_component.memory == "100Mi"
    assert test_component.gpus == 1
    assert test_component.ephemeral is None
    assert test_component.custom_resources is None


def test_torch_component_decorator(test_function_and_task_inputs):
    """Tests of Component constructor."""

    test_function, test_task_inputs = test_function_and_task_inputs

    test_component_factory = as_torch_ddp_component(test_function)

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        test_component = (
            test_component_factory(
                n_nodes=2,
                min_nodes=2,
                nproc_per_node=2,
                name="test_name",
                **test_task_inputs
            )
            .set_cpu(0.5)
            .set_memory("100Mi")
            .set_gpus(1)
        )

    # validate addition of component to pipeline context
    assert test_component == _pipeline_context.components[0]

    # validate component attributes
    assert isinstance(test_component, TorchDDPComponent)
    assert test_component.implementation == "torch-ddp"
    assert test_component.base_name == "test-name"
    assert test_component.name == "test-name-0"
    assert test_component.func == test_function
    assert test_component.hera_template_kwargs == {}
    assert test_component.n_nodes == 2
    assert test_component.min_nodes == 2
    assert test_component.nproc_per_node == 2
    assert test_component.cpu == 0.5
    assert test_component.memory == "100Mi"
    assert test_component.gpus == 1
    assert test_component.ephemeral is None
    assert test_component.custom_resources is None


def test_parameter_torch_component_to_hera(
    test_output_dir, test_mock_pipeline
):
    """Declaration of Component using InputParameter and OutputParameter"""

    # mock active pipeline with 2 inputs
    pipeline_input_a = InputParameter(name="a", value=1)
    pipeline_input_a.set_owner(test_mock_pipeline)
    pipeline_input_b = InputParameter(name="b", value=2)
    pipeline_input_b.set_owner(test_mock_pipeline)

    with _pipeline_context:
        print(_pipeline_context)
        _pipeline_context.clear()

        # add components to pipeline context
        a_plus_b = (
            add_parameters_torch_ddp_factory(
                "a_plus_b",
                n_nodes=2,
                min_nodes=1,
                nproc_per_node=5,
                a=pipeline_input_a,
                b=pipeline_input_b,
            )
            .set_cpu(1)
            .set_memory("1Gi")
        )

        a_plus_b_plus_2 = (
            add_parameters_torch_ddp_factory(
                "a_plus_b_plus_2",
                n_nodes=2,
                min_nodes=2,
                nproc_per_node=4,
                a=a_plus_b.outputs["sum"],
                b=InputParameter("two", 2),
            )
            .set_gpus(1)
            .set_ephemeral("1Ti")
        )

    a_plus_b.task_factory = a_plus_b.build_hera_task_factory()
    a_plus_b_plus_2.task_factory = a_plus_b_plus_2.build_hera_task_factory()

    with WorkflowTemplate(
        name="test-parameter-torch-ddp-component-workflow-template",
        entrypoint="test_dag",
        namespace="argo",
        arguments=[
            Parameter(name="a", value=1),
            Parameter(name="b", value=2),
        ],
    ) as wft:

        a_plus_b.service_templates = a_plus_b.build_service_templates()
        a_plus_b_plus_2.service_templates = (
            a_plus_b_plus_2.build_service_templates()
        )

        with DAG(name="test_dag"):
            a_plus_b.to_hera()
            a_plus_b_plus_2.to_hera()

    wft.to_file(test_output_dir)

    task_names = [task.name for task in wft.templates[4].tasks]
    assert task_names == [
        "a-plus-b-create-torch-ddp-service",
        "a-plus-b-0",
        "a-plus-b-0-worker-1",
        "a-plus-b-delete-torch-ddp-service",
        "a-plus-b-plus-2-create-torch-ddp-service",
        "a-plus-b-plus-2-0",
        "a-plus-b-plus-2-0-worker-1",
        "a-plus-b-plus-2-delete-torch-ddp-service",
    ]

    script_template_names = [template.name for template in wft.templates]
    assert script_template_names == [
        "a-plus-b-create-torch-ddp-service",
        "a-plus-b-delete-torch-ddp-service",
        "a-plus-b-plus-2-create-torch-ddp-service",
        "a-plus-b-plus-2-delete-torch-ddp-service",
        "test_dag",
        "a-plus-b-0",
        "a-plus-b-1",
        "a-plus-b-plus-2-0",
        "a-plus-b-plus-2-1",
    ]

    assert wft.templates[5].labels["torch-node"] == "0"
    assert wft.templates[5].labels["torch-job"] == "a-plus-b-0"
    assert wft.templates[6].labels["torch-node"] == "1"

    assert wft.templates[7].labels["torch-node"] == "0"
    assert (
        wft.templates[7].labels["torch-job"] == "a-plus-b-plus-2-0"
    )  # noqa: E501
    assert wft.templates[8].labels["torch-node"] == "1"


def test_artifact_torch_component_to_hera(
    test_output_dir,
    test_mock_pipeline,
):

    # mock active pipeline with 2 inputs
    pipeline_input_a = InputParameter(name="a", value=1)
    pipeline_input_a.set_owner(test_mock_pipeline)

    with _pipeline_context:
        _pipeline_context.clear()

        # add components to pipeline context
        convert = (
            convert_to_artifact_torch_ddp_factory(
                "convert_parameters",
                n_nodes=2,
                min_nodes=1,
                nproc_per_node=5,
                a=pipeline_input_a,
            )
            .set_cpu(0.8)
            .set_memory("2Pi")
        )

        show = (
            show_parameter_torch_ddp_factory(
                "show_artifacts",
                a=convert.outputs["a_art"],
            )
            .set_gpus(2)
            .set_ephemeral("10Ki")
        )

    convert.task_factory = convert.build_hera_task_factory()
    show.task_factory = show.build_hera_task_factory()

    with WorkflowTemplate(
        name="test-artifact-torch-ddp-component-workflow-template",
        entrypoint="test_dag",
        namespace="argo",
        arguments=[
            Parameter(name="a", value=1),
        ],
    ) as wft:

        convert.service_templates = convert.build_service_templates()
        show.service_templates = show.build_service_templates()

        with DAG(name="test_dag"):
            convert.to_hera()
            show.to_hera()

    wft.to_file(test_output_dir)

    task_names = [task.name for task in wft.templates[4].tasks]
    assert task_names == [
        "convert-parameters-create-torch-ddp-service",
        "convert-parameters-0",
        "convert-parameters-0-worker-1",
        "convert-parameters-delete-torch-ddp-service",
        "show-artifacts-create-torch-ddp-service",
        "show-artifacts-0",
        "show-artifacts-delete-torch-ddp-service",
    ]

    script_template_names = [template.name for template in wft.templates]
    assert script_template_names == [
        "convert-parameters-create-torch-ddp-service",
        "convert-parameters-delete-torch-ddp-service",
        "show-artifacts-create-torch-ddp-service",
        "show-artifacts-delete-torch-ddp-service",
        "test_dag",
        "convert-parameters-0",
        "convert-parameters-1",
        "show-artifacts-0",
    ]

    assert wft.templates[5].labels["torch-node"] == "0"
    assert (
        wft.templates[5].labels["torch-job"]
        == "convert-parameters-0"  # noqa: E501
    )
    assert wft.templates[6].labels["torch-node"] == "1"

    assert wft.templates[7].labels["torch-node"] == "0"
    assert wft.templates[7].labels["torch-job"] == "show-artifacts-0"
