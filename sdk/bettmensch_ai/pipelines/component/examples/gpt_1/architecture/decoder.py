import torch.nn as nn

from .attention import MultiHeadedAttention, decoder_mask_ca, decoder_mask_sa
from .utils import (
    LayerNorm,
    PositionwiseFeedForward,
    SublayerConnection,
    clones,
)


# --- [3] DecoderLayer and Decoder
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn and feed forward (defined below)"

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, memory, src_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer: DecoderLayer, N: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask):

        src_mask = decoder_mask_sa(src_mask)
        for layer in self.layers:
            x = layer(x, memory, src_mask)
        x = self.norm(x)

        return x
