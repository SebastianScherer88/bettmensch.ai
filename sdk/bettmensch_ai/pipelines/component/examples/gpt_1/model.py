from typing import Callable, Optional, Union

import torch
from jaxtyping import Bool, Float


class VerboseIOModule(torch.nn.Module):
    """Utility Mixin to allow for toggling of the IO shape display at runtime.
    Useful for debugging model architecture."""

    _id: Optional[str] = None
    _verbose: bool = False
    _nest_level: Optional[int] = None

    def __init__(self, id: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._id = id
        # self.forward = self.display_io_sizes(id)(self.forward)

    def set_io_verbosity(self, value: bool):
        self._verbose = value

        # set on all child modules
        for module in self.modules():
            if isinstance(module, VerboseIOModule) and module != self:
                module._verbose = value

    def set_nest_level(self, level: int = 0):
        assert (
            self.nest_level is None
        ), f"Nest level {self.nest_level} has already been set and "
        f"cannot be re-set for {self.id}"
        self._nest_level = level

        # cascade to child modules in nested fashion to generate correct nest
        # level we need to skip modules that have the level set already as each
        # module has line of sight of ALL its child modules' descendants (as
        # opposed to just its own child modules)
        for module in self.modules():
            if (
                isinstance(module, VerboseIOModule)
                and (module != self)  # noqa: W503
                and (module.nest_level is None)  # noqa: W503
            ):
                module.set_nest_level(level=self.nest_level + 1)

    @property
    def id(self) -> Union[str, None]:
        return self._id

    @property
    def is_verbose(self) -> bool:
        return self._verbose

    @property
    def nest_level(self) -> int:
        return self._nest_level

    @staticmethod
    def display_io_sizes(id: Optional[str] = None):
        def wrapper(forward: Callable):
            def func(*args, **kwargs):

                parent_module_mixin = args[0]
                assert isinstance(parent_module_mixin, VerboseIOModule)
                if id is not None:
                    display_id = id
                elif parent_module_mixin.id is not None:
                    display_id = parent_module_mixin.id
                else:
                    display_id = str(type(parent_module_mixin))

                indent = parent_module_mixin.nest_level * "    "
                verbose = parent_module_mixin.is_verbose

                if verbose:
                    for i, arg in enumerate(args[1:]):
                        try:
                            print(
                                f"{indent}[{display_id}]'s positional input"
                                f" {i+1}'s size: {arg.size()}"
                            )
                        except AttributeError:
                            print(
                                f"{indent}[{display_id}]'s positional input"
                                f" {i+1} is not a torch tensor: {type(arg)}"
                            )

                    for arg_name, arg_value in kwargs.items():
                        try:
                            print(
                                f"{indent}[{display_id}]'s named input"
                                f" {arg_name}'s size: {arg_value.size()}"
                            )
                        except AttributeError:
                            continue

                result = forward(*args, **kwargs)

                if verbose:
                    try:
                        print(
                            f"{indent}[{display_id}]'s output's size:"
                            f" {result.size()}"
                        )
                    except AttributeError:
                        print(
                            f"{indent}[{display_id}]'s output is not a torch"
                            f" tensor: {type(result)}"
                        )

                return result

            return func

        return wrapper


class Embedding(VerboseIOModule):
    def __init__(
        self,
        n_vocab: int,
        n_tokens: int = 512,
        dim_embed: int = 768,
        dropout: float = 0.1,
        id: str = "Embedding",
    ):
        super().__init__(id=id)
        self.embed = torch.nn.Embedding(
            num_embeddings=n_vocab, embedding_dim=dim_embed
        )
        self.pos: Float[
            torch.Tensor, "1 n_tokens dim_embed"
        ] = torch.nn.Parameter(torch.zeros(size=(1, n_tokens, dim_embed)))
        self.dropout = torch.nn.Dropout(p=dropout)

    @VerboseIOModule.display_io_sizes()
    def forward(
        self, x: Float[torch.Tensor, "n_batch n_tokens"]
    ) -> Float[torch.Tensor, "n_batch n_tokens dim_embed"]:

        x_embedded: Float[
            torch.Tensor, "n_batch n_tokens dim_embed"
        ] = self.embed(x)
        x_pos_embedded = self.dropout(x_embedded + self.pos)

        return x_pos_embedded


def generate_padded_subsequent_mask(
    input_mask: Bool[torch.Tensor, "n_batch n_tokens"]
) -> Float[torch.Tensor, "n_batch n_tokens n_tokens"]:
    """Generates a mask that, when passed to the attention function, forces the
      coefficients to be 0 for
    - values/tokens that come after the query/token in question in the original
        input sequence
    - values/tokens that stem from padded tokens in the original input
        sequence,
    thus forcing the attention to the non-trivial set of values/tokens. This
    reduces noise during optimization, which speeds up training and improves
     model performance
    """
    n_tokens = input_mask.size()[-1]
    padded_token_mask: Float[
        torch.Tensor, "n_batch 1 n_tokens"
    ] = input_mask.unsqueeze(1)
    subsequent_token_mask: Float[torch.Tensor, "1 n_tokens n_tokens"] = (
        torch.tril(torch.ones(size=(n_tokens, n_tokens)))
        .type(torch.bool)
        .unsqueeze(0)
    )
    final_mask = padded_token_mask & subsequent_token_mask

    return final_mask


def attention(
    queries: Float[torch.Tensor, "n_batch n_queries dim_query_key"],
    keys: Float[torch.Tensor, "n_batch n_keys_values dim_query_key"],
    values: Float[torch.Tensor, "n_batch n_keys_values dim_value"],
    mask: Optional[Float[torch.Tensor, "1 n_queries n_keys_values"]],
) -> Float[torch.Tensor, "n_batch n_query dim_value"]:
    """Single head attention function.

    Projections of the multihead attention block need to be done before/after

    Example:
    q = Tensor([[[1,2],[3,4]]])
    k = Tensor([[[1,2],[3,4]]])
    v = Tensor([[[1,2,3],[4,5,6]]])

    attention(q,k,v)
    # tensor([[[3.9089, 4.9089, 5.9089],
    #      [3.9991, 4.9991, 5.9991]]])
    """
    # get dimension of value space for normalization purposes
    dim_value = torch.tensor(values.shape[-1])

    coefficients: Float[
        torch.Tensor, "n_batch n_queries n_keys_values"
    ] = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(dim_value)

    # set invalid coefficients to -inf, so that they become ~0 convex
    # coefficients after the application of the softmax
    masked_coefficients = coefficients.masked_fill(
        mask == False, -1e9  # noqa: E712
    )

    # convert pseudo coefficients into convex combination coefficients
    convex_coefficients: Float[
        torch.Tensor, "n_batch n_queries n_keys_values"
    ] = torch.softmax(masked_coefficients, dim=-1)

    # calculate attention tensor
    att: Float[torch.Tensor, "n_batch n_queries n_keys_values"] = torch.matmul(
        convex_coefficients, values
    )

    return att


class MultiHeadedAttention(VerboseIOModule):
    """Implements a multi head attention layer that can be used for cross and
    or self attention. As described in
    - https://arxiv.org/pdf/1706.03762 (originally introduced)
    - https://cdn.openai.com/research-covers/language-unsupervised/...
        ...language_understanding_paper.pdf (referenced)
    """

    def __init__(
        self,
        n_heads: int = 12,
        dim_input: int = 768,
        dropout: float = 0.1,
        id: str = "MultiHeadAttention",
    ):
        super().__init__(id=id)

        assert (
            dim_input % n_heads == 0
        ), f"Query & value embedding size {dim_input} is not divisable by "
        f"attention head count {n_heads}"

        self.n_heads = n_heads
        self.dim_input = dim_input
        self.dim_sh_embed = int(dim_input / n_heads)

        # initialize projections for queries, keys and values for all channels
        self.W_queries = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.dim_input, self.dim_sh_embed)
                for _ in range(n_heads)
            ]
        )
        self.W_keys = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.dim_input, self.dim_sh_embed)
                for _ in range(n_heads)
            ]
        )
        self.W_values = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.dim_input, self.dim_sh_embed)
                for _ in range(n_heads)
            ]
        )
        self.W_out = torch.nn.Linear(self.dim_input, self.dim_input)
        self.dropout = torch.nn.Dropout(p=dropout)

    @VerboseIOModule.display_io_sizes()
    def forward(
        self,
        x: Float[torch.Tensor, "n_batch n_queries dim_mh_embed"],
        keys: Float[torch.Tensor, "n_batch n_keys_values dim_mh_embed"],
        values: Float[torch.Tensor, "n_batch n_keys_values dim_mh_embed"],
        mask: Optional[Float[torch.Tensor, "1 n_queries n_keys_values"]],
    ) -> Float[torch.Tensor, "n_batch n_queries dim_mh_embed"]:
        """Forward pass of a generic multi head attention layer.
        Requires mask."""

        att_list = []

        for W_Q_i, W_K_i, W_V_i in zip(
            self.W_queries, self.W_keys, self.W_values
        ):
            queries_i = W_Q_i(x)
            keys_i = W_K_i(keys)
            values_i = W_V_i(values)

            att_i = attention(
                queries=queries_i, keys=keys_i, values=values_i, mask=mask
            )
            att_list.append(att_i)

        att = torch.concatenate(att_list, -1)

        multi_head_attention = self.dropout(self.W_out(att))

        return multi_head_attention


class FeedForward(VerboseIOModule):
    """The feed forward layer (including dropout) as described in https://...
    ...cdn.openai.com/research-covers/language-unsupervised/...
    ...language_understanding_paper.pdf. Defaults to the exact configuration
     presented in the paper."""

    def __init__(
        self,
        dim_input: int = 768,
        dim_ff: int = 3072,
        dropout: float = 0.1,
        activation_class=torch.nn.GELU,
        id: str = "FeedForward",
    ):
        """By default, the GELU activation is used as described in the
        https://cdn.openai.com/research-covers/language-unsupervised/...
        ...language_understanding_paper.pdf paper.
        Choose nn.ReLU to implement the version described in the original
        transformer paper https://arxiv.org/pdf/1706.03762.

        Args:
            dim_input (int): The dimension of the input embedding vector space.
            dim_ff (int): The dimension of the hidden layer of this module.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            activation_class (_type_, optional): Activation function class.
                Defaults to nn.GELU.
        """

        super().__init__(id=id)

        self.dim_input = dim_input
        self.dim_ff = dim_ff

        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(dim_input, dim_ff),
            activation_class(),
            torch.nn.Linear(dim_ff, dim_input),
            torch.nn.Dropout(p=dropout),
        )

    @VerboseIOModule.display_io_sizes()
    def forward(
        self, x: Float[torch.Tensor, "n_batch n_tokens dim_input"]
    ) -> Float[torch.Tensor, "n_batch n_tokens dim_input"]:
        return self.sequential(x)


class SkipNorm(VerboseIOModule):
    """The skip + normalization layer (including dropout) wrapper as described
    in https://cdn.openai.com/research-covers/language-unsupervised/...
    ...language_understanding_paper.pdf"""

    def __init__(
        self,
        skipped_layer: Union[MultiHeadedAttention, FeedForward],
        dropout: float = 0.1,
        id: str = "SkipNorm",
    ):
        """_summary_

        Args:
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super().__init__(id=id)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.skipped_layer = skipped_layer
        self.norm_layer = torch.nn.LayerNorm(
            normalized_shape=(skipped_layer.dim_input,)
        )

    @VerboseIOModule.display_io_sizes()
    def forward(
        self,
        x: Float[torch.Tensor, "n_batch n_tokens n_dim_input"],
        *skipped_layer_args,
        **skipped_layer_kwargs,
    ) -> Float[torch.Tensor, "n_batch n_tokens n_dim_input"]:

        return self.dropout(
            self.norm_layer(
                x
                + self.skipped_layer(  # noqa: W503
                    x, *skipped_layer_args, **skipped_layer_kwargs
                )
            )
        )


class DecoderLayer(VerboseIOModule):
    def __init__(
        self,
        n_heads: int = 12,
        dim_input: int = 768,
        dropout: float = 0.1,
        id: str = "DecoderLayer",
    ):
        super().__init__(id=id)
        self.skip_norm_mha = SkipNorm(
            MultiHeadedAttention(
                n_heads=n_heads, dim_input=dim_input, dropout=dropout, id="mha"
            ),
            id="skip_mha",
        )
        self.skip_norm_ff = SkipNorm(
            FeedForward(dim_input=dim_input, dropout=dropout, id="ff"),
            id="skip_ff",
        )

    @VerboseIOModule.display_io_sizes()
    def forward(
        self,
        x: Float[torch.Tensor, "n_batch n_tokens dim_input"],
        mask: Bool[torch.Tensor, "n_batch n_tokens"],
    ) -> Float[torch.Tensor, "n_batch n_tokens dim_input"]:
        mha_mask = generate_padded_subsequent_mask(mask)
        mha_out = self.skip_norm_mha(x=x, keys=x, values=x, mask=mha_mask)
        ff_out = self.skip_norm_ff(mha_out)

        # print(f"Decoder layer output shape: {ff_out.size()}")

        return ff_out


class Decoder(VerboseIOModule):
    def __init__(
        self,
        dim_input: int = 768,
        n_decoder_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        id: str = "Decoder",
    ):
        super().__init__(id=id)

        self.layers = torch.nn.ModuleList(
            [
                DecoderLayer(
                    n_heads=n_heads,
                    dim_input=dim_input,
                    dropout=dropout,
                    id=f"DecoderLayer_{i+1}",
                )
                for i in range(n_decoder_layers)
            ]
        )

    @VerboseIOModule.display_io_sizes()
    def forward(
        self,
        x: Float[torch.Tensor, "n_batch n_tokens dim_input"],
        mask: Bool[torch.Tensor, "n_batch n_tokens"],
    ) -> Float[torch.Tensor, "n_batch n_tokens dim_input"]:
        for layer in self.layers:
            x = layer(x, mask)

        # print(f"Decoder output shape: {x.size()}")

        return x


class GPT1Core(VerboseIOModule):
    def __init__(
        self,
        n_vocab: int,
        n_tokens=512,
        dim_embed: int = 768,
        n_decoder_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        id: str = "GPT1Core",
    ):
        super().__init__(id=id)
        self.embedding = Embedding(
            n_vocab=n_vocab,
            n_tokens=n_tokens,
            dim_embed=dim_embed,
            dropout=dropout,
        )
        self.decoder = Decoder(
            dim_input=dim_embed,
            n_decoder_layers=n_decoder_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.set_nest_level()

    @VerboseIOModule.display_io_sizes()
    def forward(
        self,
        x: Float[torch.Tensor, "n_batch n_tokens"],
        mask: Bool[torch.Tensor, "n_batch n_tokens"],
    ) -> Float[torch.Tensor, "n_batch n_tokens dim_input"]:
        e = self.embedding(x)
        d = self.decoder(e, mask)

        return d


class GPT1Pretrain(GPT1Core):
    def __init__(
        self,
        n_vocab: int,
        n_tokens=512,
        dim_embed: int = 768,
        n_decoder_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        id: str = "GPT1Pretrain",
    ):
        super().__init__(
            n_vocab=n_vocab,
            n_tokens=n_tokens,
            dim_embed=dim_embed,
            n_decoder_layers=n_decoder_layers,
            n_heads=n_heads,
            dropout=dropout,
            id=id,
        )

    @VerboseIOModule.display_io_sizes()
    def forward(
        self,
        x: Float[torch.Tensor, "n_batch n_tokens dim_input"],
        mask: Bool[torch.Tensor, "n_batch n_tokens"],
    ) -> Float[torch.Tensor, "n_batch n_tokens n_vocab"]:
        """We extend the core class' forward method by remapping the last
        decoder layer's outputs to the vocabulary space by multiplying with the
        transpose of the initial embedding matrix (and taking the (log-)
        softmax).

        This output format allows for easier and more stable calculation of
        pretraining loss, and isnt needed during finetuning or inference. Since
        the log is a monotonous function, these outputs can also be used to
        sanity check pretraining progress by selecting the largest entry in the
        return array along the vocabuary (e.g. last) dimension and mapping its
        index back to its respective token.
        """
        d = super().forward(x, mask)
        v: Float[torch.Tensor, "n_batch n_tokens n_vocab"] = torch.matmul(
            d, self.embedding.embed.weight.transpose(-2, -1)
        )
        t = torch.log_softmax(v, -1)

        return t
