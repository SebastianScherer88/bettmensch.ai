import torch.nn as nn

from .attention import MultiHeadedAttention, encoder_mask
from .utils import (
    LayerNorm,
    PositionwiseFeedForward,
    SublayerConnection,
    clones,
)


# --- [2] EncoderLayer and Encoder
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."

        # print(f"[ENCODER] Encoder input: {x.size()}")
        # print(f"[ENCODER] Encoder mask pre transform: {mask.size()}")
        mask = encoder_mask(mask)
        # print(f"[ENCODER] Encoder mask post transform: {mask.size()}")

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        # print(f"[ENCODER] Encoder output: {x.size()}")

        return x
