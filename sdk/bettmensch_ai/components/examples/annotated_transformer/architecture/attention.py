import math

import torch
import torch.nn as nn

from .utils import clones


def attention(query, key, value, coefficient_mask=None, dropout=None):
    # Expected tensor dimensions:
    # query: # (n_batch, n_query, d_k)
    # key: # (n_batch, n_key, d_k)
    # value: # (n_batch, n_key, d_k)
    d_k = query.size(-1)
    softmax_input = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
        d_k
    )  # (n_batch, n_query, n_key)

    if coefficient_mask is not None:
        softmax_input = softmax_input.masked_fill(
            coefficient_mask == 0, -1e9
        )  # (n_batch, n_query, n_key)

    coefficient = softmax_input.softmax(dim=-1)  # (n_batch, n_query, n_key)

    if dropout is not None:
        coefficient = dropout(coefficient)  # (n_batch, n_query, n_key)

    attention = torch.matmul(coefficient, value)  # (n_batch, n_query, d_k)

    return attention, coefficient


def padded_mask(mask):
    """Create a mask to hide padding from a batch of padded sequences, src or
    tgt, that is broadcastable in the 2nd (i.e. first sequence length)
    dimension.
    Expected dimensions:
    mask: (n_batch, n_max_length), with entries 1 or 0
    """
    padded_mask = (mask != 0).unsqueeze(-2)  # (n_batch, 1, n_max_length)

    return padded_mask


def subsequent_mask(size: int):
    """Create a mask to hide subsequent positions, i.e. value vectors in an
    attention layer, that is broadcastable in the 1st (i.e. batch) dimension.
    """
    attn_shape = (1, size, size)
    not_subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )  # (1, size, size)
    subsequent_mask = not_subsequent_mask == 0

    return subsequent_mask


def encoder_mask(src_mask):
    """Create a mask to hide padding to be used in the Encoder stack's self
    attention (sub)layers.
    Expected dimensions:
    src_mask: (n_batch, n_sequence_length)
    """

    return padded_mask(src_mask)


def decoder_mask_sa(tgt_mask):
    """Create a mask to hide padding and future words (value vectors) to be
    used in the Decoder stack's self attention (sub)layers.
    Expected dimensions:
    tgt_mask: (n_batch, n_sequence_length)
    """
    tgt_length = tgt_mask.size(-1)
    dec_mask = padded_mask(tgt_mask)  # (n_batch, 1, tgt_length)
    dec_mask_sa = dec_mask & subsequent_mask(tgt_length).type_as(
        dec_mask.data
    )  # (n_batch, tgt_length, tgt_length)
    return dec_mask_sa


def decoder_mask_ca(src_mask):
    """Create a mask to hide padding to be used in the Decoder stack's cross
    attention (sub)layers.
    Expected dimensions:
    src_mask: (n_batch, n_sequence_length)
    """
    return padded_mask(src_mask)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        d_k = d_model // h
        self.d_k = d_k
        self.h = h
        self.dropout = nn.Dropout(p=dropout)
        self.head_projections = clones(clones(nn.Linear(d_model, d_k), 3), h)
        self.final_projection = nn.Linear(d_model, d_model)

        # storage attribute for visualization purposes. is re-populated for
        # each forward pass to hold the coefficient return from the `attention`
        # method.
        # Expected dimension: (n_batch, n_sequence, n_sequence, h)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        # Expected tensor dimensions:
        # query: # (n_batch, n_query, d_model)
        # key: # (n_batch, n_key, d_model)
        # value: # (n_batch, n_key, d_model)

        # initialize storage for each head's outputs
        head_attentions = []
        head_coefficients = []

        for W_q, W_k, W_v in self.head_projections:
            head_query_w = W_q(query)  # (n_batch, n_query, d_k)
            head_key_w = W_k(key)  # (n_batch, n_key, d_k)
            head_value_w = W_v(value)  # (n_batch, n_key, d_k)

            head_attention = attention(
                head_query_w, head_key_w, head_value_w, mask, self.dropout
            )  # (n_batch, n_query, d_k)
            head_attentions.append(head_attention[0])
            head_coefficients.append(head_attention[1])

        # combine individual heads' attention and coefficient outputs
        multi_attention = torch.concat(
            head_attentions, -1
        )  # (n_batch, n_query, d_model)
        coefficient = torch.stack(
            head_coefficients, -1
        )  # (n_batch, n_query, n_key, h)

        # populate coefficient storage attribute for this forward pass
        self.attn = coefficient  # (n_batch, n_query, n_key, h)

        # apply final linear projection for this layer
        projected_attention = self.final_projection(multi_attention)

        del head_query_w
        del head_key_w
        del head_value_w

        return projected_attention
