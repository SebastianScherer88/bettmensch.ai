import os

from bettmensch_ai.server import RegisteredPipeline


def test_pipeline_from_hera_artifact_workflow_template_model(
    test_output_dir,
    test_hera_artifact_workflow_template_model,
):

    test_artifact_pipeline = (
        RegisteredPipeline.from_hera_workflow_template_model(
            test_hera_artifact_workflow_template_model
        )
    )

    with open(
        os.path.join(test_output_dir, "test-artifact-server-pipeline.json"),
        "w",
    ) as file:
        file.write(test_artifact_pipeline.model_dump_json())


def test_pipeline_from_hera_parameter_workflow_template_model(
    test_output_dir,
    test_hera_parameter_workflow_template_model,
):

    test_parameter_pipeline = (
        RegisteredPipeline.from_hera_workflow_template_model(
            test_hera_parameter_workflow_template_model
        )
    )

    with open(
        os.path.join(test_output_dir, "test-parameter-server-pipeline.json"),
        "w",
    ) as file:
        file.write(test_parameter_pipeline.model_dump_json())


def test_pipeline_from_hera_torch_gpu_workflow_template_model(
    test_output_dir,
    test_hera_torch_gpu_workflow_template_model,
):

    test_torch_gpu_pipeline = (
        RegisteredPipeline.from_hera_workflow_template_model(
            test_hera_torch_gpu_workflow_template_model
        )
    )

    with open(
        os.path.join(test_output_dir, "test-torch-gpu-server-pipeline.json"),
        "w",
    ) as file:
        file.write(test_torch_gpu_pipeline.model_dump_json())
