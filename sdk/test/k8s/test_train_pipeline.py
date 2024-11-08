import pytest
from bettmensch_ai.pipelines.examples import (
    train_transformer_pipeline_1n_1p,
    train_transformer_pipeline_2n_2p,
)


@pytest.mark.train_transformer
@pytest.mark.single_node_single_gpu
def test_train_transformer_pipeline_1n_1p_decorator_and_register_and_run(
    test_output_dir, test_namespace
):
    train_transformer_pipeline_1n_1p.export(test_output_dir)

    assert not train_transformer_pipeline_1n_1p.registered
    assert train_transformer_pipeline_1n_1p.registered_id is None
    assert train_transformer_pipeline_1n_1p.registered_name is None
    assert train_transformer_pipeline_1n_1p.registered_namespace is None

    train_transformer_pipeline_1n_1p.register()

    assert train_transformer_pipeline_1n_1p.registered
    assert train_transformer_pipeline_1n_1p.registered_id is not None
    assert train_transformer_pipeline_1n_1p.registered_name.startswith(
        f"pipeline-{train_transformer_pipeline_1n_1p.name}-"
    )
    assert (
        train_transformer_pipeline_1n_1p.registered_namespace == test_namespace
    )  # noqa: E501

    train_transformer_flow = train_transformer_pipeline_1n_1p.run(
        inputs={
            "dataset": "multi30k",
            "source_language": "de",
            "target_language": "en",
            "batch_size": 32,
            "num_epochs": 1,
            "accum_iter": 10,
            "base_lr": 1.0,
            "max_padding": 72,
            "warmup": 3000,
        },
        wait=True,
    )

    assert train_transformer_flow.status.phase == "Succeeded"


@pytest.mark.train_transformer
@pytest.mark.multi_node_multi_gpu
def test_train_transformer_pipeline_2n_2p_decorator_and_register_and_run(
    test_output_dir, test_namespace
):
    train_transformer_pipeline_2n_2p.export(test_output_dir)

    assert not train_transformer_pipeline_2n_2p.registered
    assert train_transformer_pipeline_2n_2p.registered_id is None
    assert train_transformer_pipeline_2n_2p.registered_name is None
    assert train_transformer_pipeline_2n_2p.registered_namespace is None

    train_transformer_pipeline_2n_2p.register()

    assert train_transformer_pipeline_2n_2p.registered
    assert train_transformer_pipeline_2n_2p.registered_id is not None
    assert train_transformer_pipeline_2n_2p.registered_name.startswith(
        f"pipeline-{train_transformer_pipeline_2n_2p.name}-"
    )
    assert (
        train_transformer_pipeline_2n_2p.registered_namespace == test_namespace
    )  # noqa: E501

    train_transformer_flow = train_transformer_pipeline_2n_2p.run(
        inputs={
            "dataset": "multi30k",
            "source_language": "de",
            "target_language": "en",
            "batch_size": 32,
            "num_epochs": 1,
            "accum_iter": 10,
            "base_lr": 1.0,
            "max_padding": 72,
            "warmup": 3000,
        },
        wait=True,
    )

    assert train_transformer_flow.status.phase == "Succeeded"
