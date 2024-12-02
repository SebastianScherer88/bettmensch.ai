from enum import Enum
from typing import List

from pydantic import BaseModel, validator
from torchtext import datasets


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = src != pad
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.tgt != pad
            self.ntokens = self.tgt_mask.data.sum()


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class ListEnum(Enum):
    @classmethod
    def list(cls) -> List[str]:
        return [a.value for a in cls]


class SpecialTokens(ListEnum):
    start: str = "<s>"
    end: str = "</s>"
    blank: str = "<blank>"
    unk: str = "<unk>"


class SupportedDatasets(ListEnum):
    iwslt2016: str = "iwslt2016"
    iwslt2017: str = "iwslt2017"
    multi30k: str = "multi30k"


TRANSLATION_DATASETS = {
    SupportedDatasets.iwslt2016.value: datasets.IWSLT2016,
    SupportedDatasets.iwslt2017.value: datasets.IWSLT2017,
    SupportedDatasets.multi30k.value: datasets.Multi30k,
}


class SupportedLanguages(ListEnum):
    english: str = "en"
    french: str = "fr"
    german: str = "de"
    arabic: str = "ar"
    czech: str = "cs"
    italian: str = "it"
    romanian: str = "ro"
    dutch: str = "nl"


class TrainConfig(BaseModel):
    dataset: str = SupportedDatasets.multi30k.value
    source_language: str = SupportedLanguages.german.value
    target_language: str = SupportedLanguages.english.value
    source_tokenizer_path: str
    target_tokenizer_path: str
    vocabularies_path: str
    batch_size: int = 32
    distributed: bool = False
    num_epochs: int = 8
    accum_iter: int = 10
    base_lr: float = 1.0
    max_padding: int = 72
    warmup: int = 3000
    output_directory: str = "."

    @validator("dataset")
    def check_languages(v):
        assert v in SupportedDatasets.list()

        return v

    @validator("source_language")
    def check_source_language(v):
        assert v in SupportedLanguages.list()

        return v

    @validator("target_language")
    def check_target_language(v):
        assert v in SupportedLanguages.list()

        return v
