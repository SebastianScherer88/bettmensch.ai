from bettmensch_ai.pipelines import as_pipeline
from bettmensch_ai.pipelines.component.examples.annotated_transformer import (
    get_tokenizers_and_vocabularies_factory,
    train_transformer_factory,
)
from bettmensch_ai.pipelines.constants import ARGO_NAMESPACE, COMPONENT_IMAGE
from bettmensch_ai.pipelines.io import InputParameter


@as_pipeline("test-train-pipeline-1n", ARGO_NAMESPACE, True)
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

    get_tokenizers_and_vocabularies = (
        get_tokenizers_and_vocabularies_factory(
            "preprocess",
            hera_template_kwargs={
                "image": COMPONENT_IMAGE.annotated_transformer.value
            },
            dataset=dataset,
            source_language=source_language,
            target_language=target_language,
            max_padding=max_padding,
        )
        .set_memory("1000Mi")
        .set_cpu(1)
    )

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
        .set_cpu(3.5)  # works with 3, fails with 1
        .set_gpus(1)
        .set_memory("15Gi")  # works with 4Gi
    )


@as_pipeline("test-train-pipeline-xn", ARGO_NAMESPACE, True)
def train_transformer_pipeline_2n_1p(
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

    get_tokenizers_and_vocabularies = (
        get_tokenizers_and_vocabularies_factory(
            "preprocess",
            hera_template_kwargs={
                "image": COMPONENT_IMAGE.annotated_transformer.value
            },
            dataset=dataset,
            source_language=source_language,
            target_language=target_language,
            max_padding=max_padding,
        )
        .set_memory("1000Mi")
        .set_cpu(1)
    )

    train_transformer = (  # noqa: F841
        train_transformer_factory(
            "train-transformer",
            hera_template_kwargs={
                "image": COMPONENT_IMAGE.annotated_transformer.value
            },
            n_nodes=2,
            min_nodes=2,
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
        .set_cpu(3.5)  # works with 3, fails with 1
        .set_gpus(1)
        .set_memory("15Gi")  # works with 4Gi
    )
