"""Replication of GPT2 architecture in MLX."""

import math
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Literal, Self

import mlx.core as mx
from mlx import nn
from mlx.optimizers import Optimizer

from neurnet.arch.llm.config import LanguageModelConfig
from neurnet.arch.llm.kv_cache import KVCache, LayerCacheMLX
from neurnet.arch.llm.types import LanguageModelMLX
from neurnet.nn import MLXLossFunction, normal_like
from neurnet.utils.data import MLXDataLoader

from .config import GPT2ModelType

# ===---------------------------------------------------------------------------===
# GPT2 Model
# ===---------------------------------------------------------------------------===


class GPT2MLXModel(LanguageModelMLX):
    def __init__(
        self,
        config: LanguageModelConfig,
        *,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
    ) -> None:
        super().__init__()
        self._config = config
        self._compiled_forward: Callable[[mx.array], mx.array] | None = None
        self._compiled_cache: KVCache[mx.array] | None = None
        self._compiled_cached_forward: Callable[[mx.array], mx.array] | None = None

        self.transformer = Transformer(config, embd_pdrop, attn_pdrop, resid_pdrop)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        self._initialize_parameters()

        # Weight sharing scheme:
        # GPT2, similarly to the original 'Attention is all you need' paper
        # the token-embedding weights are shared with the language-model head
        # weights.
        self.transformer.wte.weight = self.lm_head.weight

    def _initialize_parameters(self) -> None:
        """Initialize parameters as in OpenAI's original GPT-2 implementation."""
        # NOTE: Do not apply the often-seen 0.02 / sqrt(2 * n_layers) residual-projection
        # scaling if the goal is exact OpenAI GPT-2 repository behavior;
        # that is a later convention, not what OpenAI’s original model.py does.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight = normal_like(module.weight, std=0.02, mean=0.0)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias = mx.zeros_like(module.bias)
            elif isinstance(module, nn.Embedding):
                module.weight = normal_like(module.weight, std=0.02, mean=0.0)
            elif isinstance(module, nn.LayerNorm):
                module.weight = mx.ones_like(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias = mx.zeros_like(module.bias)

        # GPT-2 alone uses a smaller standard deviation for position embeddings.
        self.transformer.wpe.weight = normal_like(
            self.transformer.wpe.weight, std=0.01, mean=0.0
        )

    def __call__(
        self,
        indices: mx.array,
        cache: KVCache[mx.array] | None = None,
    ) -> mx.array:
        if self._compiled_forward is None:
            return self._forward(indices, cache=cache)

        if cache is None:
            return self._compiled_forward(indices)

        return self._compiled_forward_for_cache(cache)(indices)

    def compile(self) -> Self:
        """Compile GPT-2 inference, including the tensor state of one KV cache.

        MLX compiled functions only accept array trees and simple constants. The
        public ``KVCache`` is a Python object, so cached decoding captures its
        mutable list of key/value array pairs as explicit input and output state.
        """
        self._compiled_forward = mx.compile(self._forward)
        self._compiled_cache = None
        self._compiled_cached_forward = None
        return self

    def _compiled_forward_for_cache(
        self, cache: KVCache[mx.array]
    ) -> Callable[[mx.array], mx.array]:
        if cache is self._compiled_cache:
            assert self._compiled_cached_forward is not None
            return self._compiled_cached_forward

        state = [cache.cache]

        @partial(mx.compile, inputs=state, outputs=state)
        def forward(indices: mx.array) -> mx.array:
            return self._forward(indices, cache=cache)

        self._compiled_cache = cache
        self._compiled_cached_forward = forward
        return forward

    def _forward(
        self,
        indices: mx.array,
        cache: KVCache[mx.array] | None = None,
    ) -> mx.array:
        return self.lm_head(self.transformer(indices, cache=cache))

    def config(self) -> LanguageModelConfig:
        return self._config

    @classmethod
    def from_pretrained(
        cls, model_type: GPT2ModelType, cache_dir: str
    ) -> "GPT2MLXModel":
        """Loads pretrained model weights from HuggingFace, in evaluation mode."""
        from transformers import GPT2LMHeadModel

        print(
            f"pre-loading GPT2 weights from pretrained model on HuggingFace: {model_type}"
        )
        config = model_type.get_config()
        assert config.vocab_size == 50_257, "vocabulary size is always 50,257 for GPT2"
        assert config.context_length == 1024, "context length is always 1,024 for GPT2"

        model = cls(config)
        model_hf = GPT2LMHeadModel.from_pretrained(str(model_type), cache_dir=cache_dir)
        model._load_hugging_face_weights(model_hf.state_dict())
        model.eval()

        return model

    def _load_hugging_face_weights(
        self, state_dict: Mapping[str, Any]
    ) -> "GPT2MLXModel":
        _TRANSPOSED_HUGGING_FACE_WEIGHTS = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )
        weights = []
        for name, tensor in state_dict.items():
            if name.endswith((".attn.bias", ".attn.masked_bias")):
                continue

            array = tensor.numpy()
            if name.endswith(_TRANSPOSED_HUGGING_FACE_WEIGHTS):
                array = array.T

            weights.append((name, mx.array(array)))

        self.load_weights(weights)
        return self


# ===---------------------------------------------------------------------------===
# Transformer
# ===---------------------------------------------------------------------------===


class Transformer(nn.Module):
    def __init__(
        self,
        config: LanguageModelConfig,
        embd_pdrop: float,
        attn_pdrop: float,
        resid_pdrop: float,
    ) -> None:
        super().__init__()

        self.block_size = config.context_length

        # Word token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.emb_dim)
        # Word positional embeddings
        self.wpe = nn.Embedding(config.context_length, config.emb_dim)
        self.drop = nn.Dropout(embd_pdrop)
        # Hidden blocks
        self.h = [
            Block(
                config.emb_dim,
                config.n_heads,
                config.hidden_dim,
                attn_pdrop,
                resid_pdrop,
            )
            for _ in range(config.n_layers)
        ]
        # Layer-normalisation
        self.ln_f = nn.LayerNorm(config.emb_dim)

    def __call__(
        self, indices: mx.array, cache: KVCache[mx.array] | None = None
    ) -> mx.array:
        _, sequence_length = indices.shape
        first_layer_cache = cache.get(0) if cache is not None else None
        past_length = 0 if first_layer_cache is None else first_layer_cache[0].shape[2]
        total_length = past_length + sequence_length
        if total_length > self.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {total_length}; "
                f"block size is {self.block_size}"
            )

        positions = mx.arange(past_length, total_length)
        token_embeddings = self.wte(indices)
        position_embeddings = self.wpe(positions)
        x = self.drop(token_embeddings + position_embeddings)

        for layer_idx, block in enumerate(self.h):
            layer_cache = cache.get(layer_idx) if cache is not None else None
            x, new_layer_cache = block(x, cache=layer_cache)
            if cache is not None:
                cache.update(layer_idx, new_layer_cache)

        logits = self.ln_f(x)
        return logits


# ===---------------------------------------------------------------------------===
# Transformer Block
# ===---------------------------------------------------------------------------===


class Block(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        n_head: int,
        hidden_dim: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ) -> None:
        super().__init__()

        # Pre- layer normalisation
        self.ln_1 = nn.LayerNorm(dim_embed)
        # Attention block
        self.attn = CausalSelfAttention(
            dim_embed,
            n_head,
            attn_pdrop,
            resid_pdrop,
        )
        # Pre-MLP layer normalisation
        self.ln_2 = nn.LayerNorm(dim_embed)
        # Multi-layer perceptron (feed forward layer)
        self.mlp = FeedForward(dim_embed, hidden_dim, resid_pdrop)

    def __call__(
        self, x: mx.array, cache: LayerCacheMLX | None = None
    ) -> tuple[mx.array, LayerCacheMLX]:
        # Residual blocks: x + ...
        attention_output, new_cache = self.attn(self.ln_1(x), cache=cache)
        x = x + attention_output
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


# ===---------------------------------------------------------------------------===
# CausalSelfAttention
# ===---------------------------------------------------------------------------===


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ) -> None:
        super().__init__()
        if dim_embed % n_head != 0:
            raise ValueError("dim_embed must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = dim_embed // n_head

        # Key, query, value projections for all heads, but in a batch.
        self.c_attn = nn.Linear(dim_embed, 3 * dim_embed)
        # Output projection
        self.c_proj = nn.Linear(dim_embed, dim_embed)

        # Dropout for less overfitting
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def __call__(
        self, x: mx.array, cache: LayerCacheMLX | None = None
    ) -> tuple[mx.array, LayerCacheMLX]:
        """
        Applies causal self-attention to the input array.

        Arguments:
            x: Input array. Its expected shape is (B, T, C), where B is the batch
                dimension, T is the sequence length and C are the channels, or else,
                the embedding components.

        Returns:
            The array with attention applied to it. The return array has the same
            shape as the input array.
        """
        batch_size, sequence_length, dim_embed = x.shape

        query, key, value = mx.split(self.c_attn(x), 3, axis=-1)

        # To account for multihead attention, for the query, key and value matrices,
        # we break the embeddings in n_head parts, each one with head_dim components,
        # and transpose them so that each head is a 'batch' dimension and we operate
        # on all (B, n_head) matrices (T, head_dim) in parallel.
        # (B, T, C) -> (B, n_head, T, head_dim)
        query = query.reshape(
            batch_size,
            sequence_length,
            self.n_head,
            self.head_dim,
        ).transpose(0, 2, 1, 3)
        key = key.reshape(
            batch_size,
            sequence_length,
            self.n_head,
            self.head_dim,
        ).transpose(0, 2, 1, 3)
        value = value.reshape(
            batch_size,
            sequence_length,
            self.n_head,
            self.head_dim,
        ).transpose(0, 2, 1, 3)

        past_length = 0
        if cache is not None:
            cached_key, cached_value = cache
            past_length = cached_key.shape[2]
            key = mx.concatenate([cached_key, key], axis=2)
            value = mx.concatenate([cached_value, value], axis=2)

        # Compute the scaled self-attention between the current queries and all keys.
        attention = (query @ key.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(
            past_length + sequence_length,
            dtype=attention.dtype,
        )[past_length:, :]
        attention = mx.softmax(attention + causal_mask, axis=-1)
        attention = self.attn_dropout(attention)

        # This is effectively a weighted sum of the attentions.
        # We multiply each matrics in the last two dimensions.
        # Shape: (B, n_heads, T, total_tokens) x (B, n_heads, total_tokens, head_dim)
        # -> (B, n_heads, T, head_dim)
        output = attention @ value
        output = output.transpose(0, 2, 1, 3).reshape(
            batch_size,
            sequence_length,
            dim_embed,
        )
        output = self.c_proj(output)
        return self.resid_dropout(output), (key, value)


# ===---------------------------------------------------------------------------===
# Feed-forward network
# ===---------------------------------------------------------------------------===


class FeedForward(nn.Module):
    def __init__(self, dim_embed: int, hidden_dim: int, resid_pdrop: float) -> None:
        super().__init__()

        self.c_fc = nn.Linear(dim_embed, hidden_dim)
        # Use approximate GELU to follow GPT2 implementation closer
        self.act = nn.GELU(approx="tanh")
        self.c_proj = nn.Linear(hidden_dim, dim_embed)
        self.dropout = nn.Dropout(resid_pdrop)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


# ===---------------------------------------------------------------------------===
# Train GPT2 Model
# ===---------------------------------------------------------------------------===


def train_gpt2(
    model: GPT2MLXModel,
    dataloader: MLXDataLoader,
    optimizer: Optimizer,
    loss_fn: MLXLossFunction,
    epochs: int = 50,
    verbose: bool = True,
) -> GPT2MLXModel:

    def loss_fn_inner(m: nn.Module, ins: mx.array, targs: mx.array) -> mx.array:
        logits = m(ins)  # (batch-size, token-sequence-length, vocab-size)
        loss = mx.mean(loss_fn(logits, targs))
        return loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn_inner)

    for e in range(epochs):
        dataloader.reset()
        model.train()

        for inputs, targets in dataloader:
            _, grads = loss_and_grad_fn(model, inputs, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        if verbose:
            epoch_loss = evaluate_gpt2(model, dataloader, loss_fn)
            print(f"[Epoch {e + 1} / {epochs}]: Loss {epoch_loss.item():.4f}")

    dataloader.reset()

    return model


def evaluate_gpt2(
    model: GPT2MLXModel, dataloader: MLXDataLoader, loss_fn: MLXLossFunction
) -> mx.array:
    total_loss = mx.array(0.0)
    total_tokens = 0

    model.eval()
    dataloader.reset()

    for inputs, targets in dataloader:
        logits = model(inputs)
        losses = loss_fn(logits, targets)
        total_loss += mx.sum(losses)
        total_tokens += targets.size

    dataloader.reset()

    if total_tokens == 0:
        raise ValueError("cannot evaluate an empty dataloader.")

    return total_loss / total_tokens
