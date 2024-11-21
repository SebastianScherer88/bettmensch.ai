import torch.nn as nn
from torch.nn.functional import log_softmax


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model: int, vocab_size: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # print(f"[GENERATOR] Generator input: {x.size()}")

        x = log_softmax(self.proj(x), dim=-1)

        # print(f"[GENERATOR] Generator output: {x.size()}")

        return x
