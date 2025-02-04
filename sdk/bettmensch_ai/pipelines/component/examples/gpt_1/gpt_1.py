from bettmensch_ai.pipelines.component import (
    as_component,
    as_torch_ddp_component,
)
from bettmensch_ai.pipelines.io import (
    InputArtifact,
    InputParameter,
    OutputArtifact,
)
from train import contiguous_tokenize_to_fixed_length, pretrain, size_in_gb


def get_source_data_split(
    data_split: InputParameter = "train",  # "train" / "validation"
    data_out: OutputArtifact = None,
):
    from datasets import load_dataset

    data = load_dataset(
        "bookcorpus/bookcorpus", split=data_split, trust_remote_code=True
    )
    data.save_to_disk(data_out.path)


def get_tokenized_data_split_and_tokenizer(
    source_data_split: InputArtifact,
    n_observations: InputParameter = -1,
    sequence_length: InputParameter = 512,
    unk_token: InputParameter = "<unk>",
    bos_token: InputParameter = "<s>",  # only needed for fine-tuning tasks
    eos_token: InputParameter = "<e>",  # only needed for fine-tuning tasks
    sep_token: InputParameter = "<$>",  # only needed for fine-tuning tasks
    pad_token: InputParameter = "<p>",  # only needed for fine-tuning tasks
    batch_size: InputParameter = 50,
    display_step: InputParameter = 2000,
    tokenized_data_out: OutputArtifact = None,
    tokenizer_out: OutputArtifact = None,
):

    import pickle

    from datasets import Dataset
    from transformers import OpenAIGPTConfig, OpenAIGPTTokenizerFast

    data = Dataset.load_from_disk(source_data_split.path)

    print(f"Number of observations in data set: {len(data)}")
    print(f"Size of data set in memory (in GB): {size_in_gb(data)}")

    config = OpenAIGPTConfig()
    tokenizer = OpenAIGPTTokenizerFast.from_pretrained(config.model_type)
    tokenizer.add_special_tokens(
        {
            "unk_token": unk_token,
            "bos_token": bos_token,  # only needed for fine-tuning tasks
            "eos_token": eos_token,  # only needed for fine-tuning tasks
            "sep_token": sep_token,  # only needed for fine-tuning tasks
            "pad_token": pad_token,  # only needed for fine-tuning tasks
        }
    )

    tokenized_data = contiguous_tokenize_to_fixed_length(
        data[:n_observations]["text"],
        tokenizer=tokenizer,
        batch_size=batch_size,
        length=sequence_length,
        display_step=display_step,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    print(
        f"Number of observations in tokenized data set: {len(tokenized_data)}"
    )
    print(
        "Size of tokenized data set in memory (in GB): "
        f"{size_in_gb(tokenized_data)}"
    )

    with open(tokenized_data_out.path, "wb") as token_file:
        pickle.dump(tokenized_data, token_file)

    tokenizer.save_pretrained(tokenizer_out.path)


def pretrain_checkpoints(
    tokenized_train: InputArtifact,
    tokenized_validation: InputArtifact,
    tokenizer: InputArtifact,
    dim_embed: InputParameter,
    n_decoder_layers: InputParameter,
    n_heads: InputParameter,
    dropout: InputParameter,
    n_epochs: InputParameter,
    batch_size: InputParameter,
    shuffle: InputParameter,
    display_step: InputParameter,
    verbose: InputParameter,
):
    pretrain(
        train_data_path=tokenized_train.path,
        validation_data_path=tokenized_validation.path,
        tokenizer_path=tokenizer.path,
        dim_embed=dim_embed,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads,
        dropout=dropout,
        n_epochs=n_epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        display_step=display_step,
        verbose=verbose,
    )


get_tokenized_data_split_and_tokenizer_factory = as_component(
    get_tokenized_data_split_and_tokenizer
)

pretrain_gpt_1_factory = as_torch_ddp_component(pretrain_checkpoints)
