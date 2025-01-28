import copy

import torch
import torch.nn as nn

from .attention import MultiHeadedAttention
from .decoder import Decoder, DecoderLayer
from .embedding import Embeddings, PositionalEncoding
from .encoder import Encoder, EncoderLayer
from .generator import Generator
from .utils import PositionwiseFeedForward

# --- [5] EncoderDecorder Model


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Sequential,
        tgt_embed: nn.Sequential,
        generator: Generator,
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(
        self, src, tgt, src_mask, tgt_mask, generate_last_only: bool = False
    ):
        """Take in and process masked src and target sequences to:
        - generate encodings for all unmasked elements in input sequence
        - generate decodings for all unmasked elements of tgt sequence
        """
        # print(f"Target: {tgt.size()}")
        # print(f"Target mask: {tgt_mask.size()}")
        encoder_out = self.encode(src, src_mask)
        # print(f"Encoder output: {encoder_out.size()}")
        decoder_out = self.decode(encoder_out, src_mask, tgt, tgt_mask)
        # print(f"Decoder output: {decoder_out.size()}")
        generator_out = self.generate(
            decoder_out, generate_last_only=generate_last_only
        )

        return generator_out

    def generate(self, decoder_out, generate_last_only: bool = True):
        """Generates probability distributions over target vocabulary for all
        (unmasked) elements of the decoder representation of the tgt
        sequence"""

        # for greedy decoding during inference it will save time to only apply
        # the generator to the last, newly generated decoder representation
        if generate_last_only:
            decoder_out = decoder_out[:, -1]

        generator_out = self.generator(decoder_out)
        # print(f"Generator output: {generator_out.size()}")
        return generator_out

    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        """Implements greedy decoding by repeatedly decoding + generating
        output tokens and appending them to target sequence in subsequent
        forward pass."""
        memory = self.encode(src, src_mask)
        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            # update mask for increased ys sequence
            ys_mask = torch.ones(ys.size()).type_as(src.data)
            # generate decoded representations for each element of the ys
            # sequence
            out = self.decode(memory, src_mask, ys, ys_mask)
            # take the last element of the decoded representations of the ys
            # sequence
            prob = self.generate(out)
            # obtain index of largest prob score in this array. this is also
            # the "index" of the corresponding token in our pseudo test target
            # vocabulary
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            # append this newly predicted token to the output ys sequence for
            # next forward pass
            ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)],
                dim=1,
            )
        return ys

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position)),
        Generator(d_model, tgt_vocab_size),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def test_model(
    src_vocab_size: int = 5,
    tgt_vocab_size: int = 6,
    N: int = 1,
    d_model: int = 4,
    d_ff: int = 8,
    h: int = 2,
    dropout: float = 0.1,
    n_out: int = 10,
):
    test_model = make_model(
        src_vocab_size, tgt_vocab_size, N, d_model, d_ff, h, dropout
    )
    test_model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4]])
    # Note that tensor dimensions during decoding inference are more dynamic in
    # the second dimension, as we keep adding tokens to the target sequence in
    # an auto-regressive way. This means that the intuitive shape (1,5,5)
    # would be incompatible in our implementation with the dynamic decoder
    # query tensor during inference with shapes (1,1,5), (1,2,5), (1,3,5), ...
    # That's why the src_mask (`test_model`) uses masks that can be broadcasted
    # along that dimension for the cross attention sublayers. Torch tensor
    # operation implementations correctly resolves this to a broadcasted mask
    # filler that is synonymous with every decoder query having access to every
    # input sequence token's encoder output.
    src_mask = torch.ones(1, 5)
    ys = torch.zeros(1, 1).type_as(src)

    print(f"[TEST] Source : {src} (size {src.size()})")
    print(f"[TEST] Source mask: {src_mask} (size {src_mask.size()})")
    print(f"[TEST] Target: {ys} (size {ys.size()})")
    print("[TEST] Starting inference...")

    ys = test_model.greedy_decode(src, src_mask, 10, 0)

    print("Example Untrained Model Prediction:", ys)


def run_model_test(n_test: int = 10, n_out: int = 10):
    for _ in range(n_test):
        print("-----------------------")
        test_model(n_out=n_out)
