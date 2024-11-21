import os
from typing import Dict, List, Tuple

import spacy
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import ShardingFilter
from torch.utils.data.distributed import DistributedSampler
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import Vocab, build_vocab_from_iterator

from .utils import TRANSLATION_DATASETS, SpecialTokens, SupportedLanguages


class Preprocessor(object):

    language_src: str
    language_tgt: str
    train: ShardingFilter
    val: ShardingFilter
    test: ShardingFilter
    tokenizer_src: spacy.language.Language
    tokenizer_tgt: spacy.language.Language
    vocab_src: Vocab
    vocab_tgt: Vocab
    vocab_src_size: int
    vocab_tgt_size: int
    tokenizer_map: Dict[str, str] = {
        SupportedLanguages.english.value: "en_core_web_sm",
        SupportedLanguages.german.value: "de_core_news_sm",
    }

    def __init__(
        self,
        data_splits: Tuple[ShardingFilter, ShardingFilter, ShardingFilter],
        language_src: str = SupportedLanguages.german.value,
        language_tgt: str = SupportedLanguages.english.value,
        max_padding: int = 128,
    ):
        self.train, self.val, self.test = data_splits
        self.language_src = language_src
        self.language_tgt = language_tgt
        self.max_padding = max_padding

    def download_tokenizers(self, source_path: str, target_path: str):
        """Downloads the source and target language tokenizers into the
        specified paths, then calls `load_tokenizers`.

        Args:
            language (str): The language of the tokenizer
            path (str): The target download path
        """

        for language, path in zip(
            (self.language_src, self.language_tgt), (source_path, target_path)
        ):
            os.system(
                f"python3 -m spacy download {self.tokenizer_map[language]}"
            )
            tokenizer = spacy.load(self.tokenizer_map[language])
            tokenizer.to_disk(path)

        self.load_tokenizers(source_path, target_path)

    def load_tokenizers(self, source_path: str, target_path: str):
        """Loads spacy tokenizers for the source and target language from the
        specified paths and attaches them to the two tokenizer attributes.

        Args:
            source_path (str): The path to the on disk location of the source
                language tokenizer
            target_path (str): The path to the on disk location of the target
                language tokenizer
        """

        tokenizers = []

        for path in (source_path, target_path):
            tokenizer = spacy.load(path)
            tokenizers.append(tokenizer)

        self.tokenizer_src, self.tokenizer_tgt = tokenizers

    def tokenize_src(self, text: str):
        return [tok.text for tok in self.tokenizer_src.tokenizer(text)]

    def tokenize_tgt(self, text: str):
        return [tok.text for tok in self.tokenizer_tgt.tokenizer(text)]

    @staticmethod
    def yield_tokens(
        data_iter, tokenizer: spacy.language.Language, index: int
    ):
        for from_to_tuple in data_iter:
            yield tokenizer(from_to_tuple[index])

    def build_vocabularies(self):
        """Builds source and target vocabularies and attaches them to the
        two vocabulary attributes.
        """

        print(f"Building {self.language_src} Vocabulary ...")
        vocab_src = build_vocab_from_iterator(
            self.yield_tokens(
                self.train + self.val, self.tokenize_src, index=0
            ),
            min_freq=2,
            specials=SpecialTokens.list(),
        )

        print(f"Building {self.language_tgt} Vocabulary ...")
        vocab_tgt = build_vocab_from_iterator(
            self.yield_tokens(
                self.train + self.val, self.tokenize_src, index=1
            ),
            min_freq=2,
            specials=SpecialTokens.list(),
        )

        vocab_src.set_default_index(vocab_src[SpecialTokens.unk.value])
        vocab_tgt.set_default_index(vocab_tgt[SpecialTokens.unk.value])

        self.vocab_src, self.vocab_tgt = vocab_src, vocab_tgt
        self.vocab_src_size, self.vocab_tgt_size = len(self.vocab_src), len(
            self.vocab_tgt
        )

    def save_vocabularies(self, path: str):
        """Saves the source and target vocabularies in the specified path.

        Args:
            path (str): The path of the torch artifact the vocabluaries will be
                written to.
        """

        torch.save((self.vocab_src, self.vocab_tgt), path)

    def load_vocabularies(self, path: str) -> Tuple[Vocab, Vocab]:
        """Loads the source and target vocabularies from the specified path and
        attaches them to the two vocabulary attributes.

        Args:
            path (str): The path of the torch artifact the vocabluaries will be
                loaded from.
        """
        vocab_src, vocab_tgt = torch.load(path)
        self.vocab_src, self.vocab_tgt = vocab_src, vocab_tgt
        self.vocab_src_size, self.vocab_tgt_size = len(self.vocab_src), len(
            self.vocab_tgt
        )
        print("Finished.\nVocabulary sizes:")
        print(self.vocab_src_size)
        print(self.vocab_tgt_size)

    def collate_batch(
        self,
        batch: List[Tuple[str, str]],
        device,
        max_padding=128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        bs_id = torch.tensor(
            [self.vocab_src.get_stoi()[SpecialTokens.start.value]],
            device=device,
        )  # <s> token id
        eos_id = torch.tensor(
            [self.vocab_src.get_stoi()[SpecialTokens.end.value]], device=device
        )  # </s> token id
        src_list, tgt_list = [], []
        for (_src, _tgt) in batch:
            processed_src = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        self.vocab_src(self.tokenize_src(_src)),
                        dtype=torch.int64,
                        device=device,
                    ),
                    eos_id,
                ],
                0,
            )
            processed_tgt = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        self.vocab_tgt(self.tokenize_tgt(_tgt)),
                        dtype=torch.int64,
                        device=device,
                    ),
                    eos_id,
                ],
                0,
            )
            src_list.append(
                # warning-overwrites values for negative values of padding-len
                pad(
                    processed_src,
                    (
                        0,
                        max_padding - len(processed_src),
                    ),
                    value=self.vocab_src.get_stoi()[SpecialTokens.blank.value],
                )
            )
            tgt_list.append(
                pad(
                    processed_tgt,
                    (0, max_padding - len(processed_tgt)),
                    value=self.vocab_tgt.get_stoi()[SpecialTokens.blank.value],
                )
            )

        src = torch.stack(src_list)
        tgt = torch.stack(tgt_list)
        return (src, tgt)


def create_dataloaders(
    device: int,
    dataset: str,
    source_language: str,
    target_language: str,
    max_padding: int,
    source_tokenizer_path: str,
    target_tokenizer_path: str,
    vocabularies_path: str,
    batch_size=12000,
) -> Tuple[Preprocessor, Tuple[DataLoader, DataLoader]]:

    train_iter, valid_iter, test_iter = TRANSLATION_DATASETS[dataset](
        language_pair=(source_language, target_language)
    )

    preprocessor = Preprocessor(
        [train_iter, valid_iter, test_iter],
        source_language,
        target_language,
        max_padding,
    )

    preprocessor.load_tokenizers(
        source_path=source_tokenizer_path, target_path=target_tokenizer_path
    )
    preprocessor.load_vocabularies(path=vocabularies_path)

    # def create_dataloaders(batch_size=12000):
    def collate_fn(batch):
        return preprocessor.collate_batch(
            batch,
            device,
        )

    train_iter_map = to_map_style_dataset(
        preprocessor.train
    )  # DistributedSampler needs a dataset len()
    train_sampler = DistributedSampler(train_iter_map)
    valid_iter_map = to_map_style_dataset(preprocessor.val)
    valid_sampler = DistributedSampler(valid_iter_map)

    print(f"Size of train dataset: {len(train_iter_map)}")
    print(f"Size of valid dataset: {len(valid_iter_map)}")

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return preprocessor, (train_dataloader, valid_dataloader)
