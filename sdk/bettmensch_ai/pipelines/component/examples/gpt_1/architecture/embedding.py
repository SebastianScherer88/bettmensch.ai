import math

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, max_length: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Linear(max_length, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model) + self.pos
