import torch
from bettmensch_ai.pipelines.component.examples.gpt_1.model import (  # noqa: E501
    DecoderLayer,
    GPT1Core,
    GPT1Pretrain,
    attention,
    generate_padded_subsequent_mask,
)


def test_generate_padded_subsequent_mask():

    mask = generate_padded_subsequent_mask(
        torch.tensor(
            [[True, True, False], [True, False, False]], dtype=torch.bool
        )
    )
    torch.testing.assert_close(
        mask,
        torch.tensor(
            [
                [
                    [True, False, False],
                    [True, True, False],
                    [True, True, False],
                ],
                [
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                ],
            ]
        ),
    )


def test_attention():
    q = torch.tensor([[[1, 2], [3, 4]]], dtype=float)
    k = torch.tensor([[[1, 2], [3, 4]]], dtype=float)
    v = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=float)
    mask = torch.tensor([[[True, False], [True, True]]], dtype=bool)

    torch.testing.assert_close(
        attention(q, k, v, mask),
        torch.tensor(
            [[[1.0000, 2.0000, 3.0000], [3.9991, 4.9991, 5.9991]]],
            dtype=torch.float64,
        ),
        atol=1e-5,
        rtol=1e-4,
    )


def test_verbose_io_module():
    decoder_layer = DecoderLayer(n_heads=1, dim_input=2, dropout=0.2)
    decoder_layer.set_nest_level()
    decoder_layer.set_io_verbosity(True)
    _ = decoder_layer.forward(
        x=torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float),
        mask=torch.tensor([[True, True], [True, False]]),
    )
    print(_)


def test_gpt1_core():
    gpt1_core = GPT1Core(
        n_vocab=10, n_tokens=5, dim_embed=6, n_decoder_layers=2, n_heads=3
    )
    gpt1_core.eval()
    gpt1_core.set_io_verbosity(True)
    _ = gpt1_core.forward(
        x=torch.tensor(
            [[0, 1, 2, 9, 9], [5, 6, 7, 8, 9]], dtype=int
        ),  # 9 = padding token
        mask=torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=int),
    )


def test_gpt1_pretrain():
    gpt1_pretrain = GPT1Pretrain(
        n_vocab=10, n_tokens=5, dim_embed=6, n_decoder_layers=2, n_heads=3
    )
    gpt1_pretrain.eval()
    gpt1_pretrain.set_io_verbosity(True)
    _ = gpt1_pretrain.forward(
        x=torch.tensor(
            [[0, 1, 2, 9, 9], [5, 6, 7, 8, 9]], dtype=int
        ),  # 9 = padding token
        mask=torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=int),
    )
