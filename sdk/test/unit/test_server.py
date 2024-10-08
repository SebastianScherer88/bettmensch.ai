from bettmensch_ai.server import RegisteredFlow, RegisteredPipeline


def test_pipeline_from_hera_artifact_workflow_template_model(
    test_hera_artifact_workflow_template_model,
):

    RegisteredPipeline.from_hera_workflow_template_model(
        test_hera_artifact_workflow_template_model
    )


def test_pipeline_from_hera_parameter_workflow_template_model(
    test_hera_parameter_workflow_template_model,
):

    RegisteredPipeline.from_hera_workflow_template_model(
        test_hera_parameter_workflow_template_model
    )


def test_pipeline_from_hera_torch_gpu_workflow_template_model(
    test_hera_torch_gpu_workflow_template_model,
):

    RegisteredPipeline.from_hera_workflow_template_model(
        test_hera_torch_gpu_workflow_template_model
    )


def test_pipeline_from_hera_artifact_workflow_model(
    test_hera_artifact_workflow_model,
):

    RegisteredFlow.from_hera_workflow_model(test_hera_artifact_workflow_model)


def test_pipeline_from_hera_parameter_workflow_model(
    test_hera_parameter_workflow_model,
):

    RegisteredFlow.from_hera_workflow_model(test_hera_parameter_workflow_model)


def test_pipeline_from_hera_torch_gpu_workflow_model(
    test_hera_torch_gpu_workflow_model,
):

    RegisteredFlow.from_hera_workflow_model(test_hera_torch_gpu_workflow_model)
