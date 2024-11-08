from bettmensch_ai.components import torch_ddp_component
from bettmensch_ai.constants import ARGO_NAMESPACE, COMPONENT_IMAGE
from bettmensch_ai.io import InputParameter, OutputArtifact, OutputParameter
from bettmensch_ai.pipelines import pipeline


@torch_ddp_component
def train_transformer_component(
    dataset: InputParameter = "multi30k",
    source_language: InputParameter = "de",
    target_language: InputParameter = "en",
    batch_size: InputParameter = 32,
    num_epochs: InputParameter = 8,
    accum_iter: InputParameter = 10,
    base_lr: InputParameter = 1.0,
    max_padding: InputParameter = 72,
    warmup: InputParameter = 3000,
    trained_transformer: OutputArtifact = None,
    training_config: OutputParameter = None,
):
    import os

    from bettmensch_ai.pipelines.examples.annotated_transformer.training import (  # noqa: E501
        TrainConfig,
        train_worker,
    )

    # make sure model artifact export directory exists
    if not os.path.exists(trained_transformer.path):
        os.makedirs(trained_transformer.path)

    file_prefix = os.path.join(trained_transformer.path, f"{dataset}_model_")

    config = TrainConfig(
        dataset=dataset,
        source_language=source_language,
        target_language=target_language,
        batch_size=batch_size,
        distributed=True,
        num_epochs=num_epochs,
        accum_iter=accum_iter,
        base_lr=base_lr,
        max_padding=max_padding,
        warmup=warmup,
        file_prefix=file_prefix,
    )

    train_worker(config)

    training_config.assign(config)


@pipeline("test-train-pipeline-1n-1p", ARGO_NAMESPACE, True)
def train_transformer_pipeline_1n_1p(
    dataset: InputParameter = "multi30k",
    source_language: InputParameter = "de",
    target_language: InputParameter = "en",
    batch_size: InputParameter = 32,
    num_epochs: InputParameter = 8,
    accum_iter: InputParameter = 10,
    base_lr: InputParameter = 1.0,
    max_padding: InputParameter = 72,
    warmup: InputParameter = 3000,
):

    train_transformer = (  # noqa: F841
        train_transformer_component(
            "train-transformer",
            hera_template_kwargs={
                "image": COMPONENT_IMAGE.annotated_transformer.value
            },
            n_nodes=1,
            min_nodes=1,
            nproc_per_node=1,
            dataset=dataset,
            source_language=source_language,
            target_language=target_language,
            batch_size=batch_size,
            num_epochs=num_epochs,
            accum_iter=accum_iter,
            base_lr=base_lr,
            max_padding=max_padding,
            warmup=warmup,
        )
        .set_gpus(1)
        .set_memory("700Mi")
    )


@pipeline("test-train-pipeline-2n-2p", ARGO_NAMESPACE, True)
def train_transformer_pipeline_2n_2p(
    dataset: InputParameter = "multi30k",
    source_language: InputParameter = "de",
    target_language: InputParameter = "en",
    batch_size: InputParameter = 32,
    num_epochs: InputParameter = 8,
    accum_iter: InputParameter = 10,
    base_lr: InputParameter = 1.0,
    max_padding: InputParameter = 72,
    warmup: InputParameter = 3000,
):

    train_transformer = (  # noqa: F841
        train_transformer_component(
            "train-transformer",
            hera_template_kwargs={
                "image": COMPONENT_IMAGE.annotated_transformer.value
            },
            n_nodes=2,
            min_nodes=2,
            nproc_per_node=2,
            dataset=dataset,
            source_language=source_language,
            target_language=target_language,
            batch_size=batch_size,
            num_epochs=num_epochs,
            accum_iter=accum_iter,
            base_lr=base_lr,
            max_padding=max_padding,
            warmup=warmup,
        )
        .set_gpus(2)
        .set_memory("1500Mi")
    )
