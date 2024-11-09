from bettmensch_ai.components import torch_ddp_component
from bettmensch_ai.io import (
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
)


def get_tokenizers_and_vocabularies(
    dataset: InputParameter = "multi30k",
    source_language: InputParameter = "de",
    target_language: InputParameter = "en",
    max_padding: InputParameter = 72,
    source_tokenizer: OutputArtifact = None,
    target_tokenizer: OutputArtifact = None,
    vocabularies: OutputArtifact = None,
):

    from bettmensch_ai.components.examples.annotated_transformer.training import (  # noqa: E501
        TRANSLATION_DATASETS,
        Preprocessor,
    )

    train_iter, valid_iter, test_iter = TRANSLATION_DATASETS[dataset](
        language_pair=(source_language, target_language)
    )

    preprocessor = Preprocessor(
        data_splits=[train_iter, valid_iter, test_iter],
        language_src=source_language,
        language_tgt=target_language,
        max_padding=max_padding,
    )
    preprocessor.download_tokenizers(
        source_tokenizer.path, target_tokenizer.path
    )
    preprocessor.build_vocabularies()
    preprocessor.save_vocabularies(vocabularies.path)


def train_transformer(
    dataset: InputParameter = "multi30k",
    source_language: InputParameter = "de",
    target_language: InputParameter = "en",
    source_tokenizer: InputArtifact = None,
    target_tokenizer: InputArtifact = None,
    vocabularies: InputArtifact = None,
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

    from bettmensch_ai.components.examples.annotated_transformer.training import (  # noqa: E501
        TrainConfig,
        train_worker,
    )

    # make sure model artifact export directory exists
    if trained_transformer is not None:
        if not os.path.exists(trained_transformer.path):
            os.makedirs(trained_transformer.path)

        file_prefix = os.path.join(
            trained_transformer.path, f"{dataset}_model_"
        )
    else:
        file_prefix = os.path.join(".", f"{dataset}_model_")

    config = TrainConfig(
        dataset=dataset,
        source_language=source_language,
        target_language=target_language,
        source_tokenizer_path=source_tokenizer.path,
        target_tokenizer_path=target_tokenizer.path,
        vocabularies_path=vocabularies.path,
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

    if training_config is not None:
        training_config.assign(config)


train_transformer_factory = torch_ddp_component(train_transformer)
