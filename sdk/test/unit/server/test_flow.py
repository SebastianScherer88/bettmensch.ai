import os

from bettmensch_ai.server import RegisteredFlow


def test_flow_from_hera_artifact_workflow_model(
    test_output_dir,
    test_hera_artifact_workflow_model,
):

    test_artifact_flow = RegisteredFlow.from_hera_workflow_model(
        test_hera_artifact_workflow_model
    )

    with open(
        os.path.join(test_output_dir, "test-artifact-server-flow.json"), "w"
    ) as file:
        file.write(test_artifact_flow.model_dump_json())


def test_flow_from_hera_parameter_workflow_model(
    test_output_dir,
    test_hera_parameter_workflow_model,
):

    test_parameter_flow = RegisteredFlow.from_hera_workflow_model(
        test_hera_parameter_workflow_model
    )

    with open(
        os.path.join(test_output_dir, "test-parameter-server-flow.json"), "w"
    ) as file:
        file.write(test_parameter_flow.model_dump_json())


def test_flow_from_hera_torch_gpu_workflow_model(
    test_output_dir,
    test_hera_torch_gpu_workflow_model,
):

    test_torch_gpu_flow = RegisteredFlow.from_hera_workflow_model(
        test_hera_torch_gpu_workflow_model
    )

    with open(
        os.path.join(test_output_dir, "test-torch-gpu-server-flow.json"), "w"
    ) as file:
        file.write(test_torch_gpu_flow.model_dump_json())
