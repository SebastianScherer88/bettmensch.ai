from bettmensch_ai.pipelines.component import (
    as_component,
    as_torch_ddp_component,
)
from bettmensch_ai.pipelines.io import (
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

    import os

    from bettmensch_ai.pipelines.component.examples.annotated_transformer.training import (  # noqa: E501
        TRANSLATION_DATASETS,
        Preprocessor,
    )

    print("Getting the data splits")

    train_iter, valid_iter, test_iter = TRANSLATION_DATASETS[dataset](
        language_pair=(source_language, target_language)
    )

    print("Initializing preprocessor")

    preprocessor = Preprocessor(
        data_splits=[train_iter, valid_iter, test_iter],
        language_src=source_language,
        language_tgt=target_language,
        max_padding=max_padding,
    )

    print(
        f"Downloading tokenizers and saving to {source_tokenizer.path}&"
        f"{target_tokenizer.path}"
    )

    preprocessor.download_tokenizers(
        source_tokenizer.path, target_tokenizer.path
    )

    print(f"Local files: {os.listdir()}")
    print("Building vocabularies")

    preprocessor.build_vocabularies()

    print(f"Saving vocabularies to {vocabularies.path}")

    preprocessor.save_vocabularies(vocabularies.path)

    print(f"Local files: {os.listdir()}")

    return


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
    model_checkpoints: OutputArtifact = None,
    training_config: OutputParameter = None,
):
    from bettmensch_ai.pipelines.component.examples.annotated_transformer.training import (  # noqa: E501
        TrainConfig,
        train_worker,
    )

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
        output_directory=model_checkpoints.path,
    )

    train_worker(config)

    training_config.assign(config)


get_tokenizers_and_vocabularies_factory = as_component(
    get_tokenizers_and_vocabularies
)
train_transformer_factory = as_torch_ddp_component(train_transformer)
