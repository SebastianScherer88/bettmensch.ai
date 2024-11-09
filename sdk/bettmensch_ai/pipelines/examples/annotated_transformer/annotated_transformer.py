from bettmensch_ai.components.examples.annotated_transformer import (
    get_tokenizers_and_vocabularies_factory,
    train_transformer_factory,
)
from bettmensch_ai.constants import ARGO_NAMESPACE, COMPONENT_IMAGE
from bettmensch_ai.io import InputParameter
from bettmensch_ai.pipelines import pipeline


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

    get_tokenizers_and_vocabularies = get_tokenizers_and_vocabularies_factory(
        "preprocess",
        hera_template_kwargs={
            "image": COMPONENT_IMAGE.annotated_transformer.value
        },
        dataset=dataset,
        source_language=source_language,
        target_language=target_language,
        max_padding=max_padding,
    ).set_memory("800Mi")

    train_transformer = (  # noqa: F841
        train_transformer_factory(
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
            source_tokenizer=get_tokenizers_and_vocabularies.outputs[
                "source_tokenizer"
            ],
            target_tokenizer=get_tokenizers_and_vocabularies.outputs[
                "target_tokenizer"
            ],
            vocabularies=get_tokenizers_and_vocabularies.outputs[
                "vocabularies"
            ],
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

    get_tokenizers_and_vocabularies = get_tokenizers_and_vocabularies_factory(
        "preprocess",
        hera_template_kwargs={
            "image": COMPONENT_IMAGE.annotated_transformer.value
        },
        dataset=dataset,
        source_language=source_language,
        target_language=target_language,
        max_padding=max_padding,
    ).set_memory("800Mi")

    train_transformer = (  # noqa: F841
        train_transformer_factory(
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
            source_tokenizer=get_tokenizers_and_vocabularies.outputs[
                "source_tokenizer"
            ],
            target_tokenizer=get_tokenizers_and_vocabularies.outputs[
                "target_tokenizer"
            ],
            vocabularies=get_tokenizers_and_vocabularies.outputs[
                "vocabularies"
            ],
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
