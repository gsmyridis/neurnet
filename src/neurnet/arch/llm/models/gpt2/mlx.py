"""Replication of GPT2 architecture in MLX."""

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sized
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Self

import mlx.core as mx
from mlx import nn
from mlx.optimizers import Optimizer
from mlx.utils import tree_map
from tqdm import tqdm

from neurnet.arch.llm.config import LanguageModelConfig
from neurnet.arch.llm.kv_cache import KVCacheMLX, LayerCacheMLX, fixed_cache_capacity
from neurnet.arch.llm.types import LanguageModelMLX
from neurnet.datasets import MLXDataLoader
from neurnet.nn import MLXLossFunction, normal_like

from .config import GPT2_CONFIG_124M, GPT2ModelType

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

        self.transformer = Transformer(config, embd_pdrop, attn_pdrop, resid_pdrop)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        self._initialize_parameters()

        # Weight sharing scheme:
        # GPT2, similarly to the original 'Attention is all you need' paper
        # the token-embedding weights are shared with the language-model head
        # weights.
        self.transformer.wte.weight = self.lm_head.weight

        # Compiled operations
        self._compiled_forward: Callable[[mx.array], mx.array] | None = None
        self._compiled_decode: Callable[..., Any] | None = None

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
        cache: KVCacheMLX | None = None,
    ) -> mx.array:
        if cache is None:
            if self._compiled_forward is not None:
                return self._compiled_forward(indices)
            return self._forward(indices)

        if cache.offset == 0:
            return self._prefill_cache(indices, cache)
        if indices.shape[1] != 1:
            raise ValueError("fixed-cache decoding requires one token at a time")
        if cache.offset >= min(cache.capacity, self._config.context_length):
            raise ValueError("GPT-2 context length exceeded")

        decode = self._compiled_decode or self._decode_one
        logits, layers, position = decode(indices, cache.layers, cache.position)
        cache.layers = layers
        cache.position = position
        cache.offset += 1
        mx.eval(logits, cache.layers, cache.position)

        return logits

    def compile(self) -> Self:
        """Compile uncached inference and the stable-shape one-token decoder."""
        self._compiled_forward = mx.compile(self._forward)
        self._compiled_decode = mx.compile(self._decode_one)
        return self

    def create_kv_cache(self, batch_size: int, max_length: int) -> KVCacheMLX:
        capacity = fixed_cache_capacity(max_length, self._config.context_length)
        return KVCacheMLX(
            n_layers=self._config.n_layers,
            batch_size=batch_size,
            n_heads=self._config.n_heads,
            head_dim=self._config.effective_head_dim,
            capacity=capacity,
            dtype=self.transformer.wte.weight.dtype,
        )

    def _prefill_cache(self, indices: mx.array, cache: KVCacheMLX) -> mx.array:
        if indices.shape[1] > cache.capacity:
            raise ValueError("GPT-2 context length exceeded")
        logits = self._forward(indices, cache=cache)
        cache.finish_prefill(indices.shape[1])
        mx.eval(logits, cache.layers, cache.position)
        return logits

    def _decode_one(
        self,
        indices: mx.array,
        layers: tuple[LayerCacheMLX, ...],
        position: mx.array,
    ) -> tuple[mx.array, tuple[LayerCacheMLX, ...], mx.array]:
        hidden, updated_layers = self.transformer.decode_one(indices, layers, position)
        return self.lm_head(hidden), updated_layers, position + 1

    def _forward(
        self,
        indices: mx.array,
        cache: KVCacheMLX | None = None,
    ) -> mx.array:
        return self.lm_head(self.transformer(indices, cache=cache))

    def config(self) -> LanguageModelConfig:
        return self._config

    # ===-------------------------------------------------------------------===
    # Load model
    # ===-------------------------------------------------------------------===

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: LanguageModelConfig = GPT2_CONFIG_124M,
    ) -> Self:
        """Load native MLX weights in evaluation mode, preserving their dtype.

        Accepts ``.safetensors`` and ``.npz`` files written by ``save_weights``.
        Weight-only checkpoints require the matching architecture configuration;
        the default matches the 124M model used by the training script.
        """
        path = Path(checkpoint_path).expanduser()
        if path.suffix not in (".safetensors", ".npz"):
            raise ValueError("checkpoint path must end with '.safetensors' or '.npz'")
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint file not found: {path}")

        model = cls(config)
        model.load_weights(str(path), strict=True)
        model.transformer.wte.weight = model.lm_head.weight
        model.eval()
        return model

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

    def __call__(self, indices: mx.array, cache: KVCacheMLX | None = None) -> mx.array:
        _, sequence_length = indices.shape
        if sequence_length > self.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {sequence_length}; "
                f"block size is {self.block_size}"
            )

        positions = mx.arange(sequence_length)
        token_embeddings = self.wte(indices)
        position_embeddings = self.wpe(positions)
        x = self.drop(token_embeddings + position_embeddings)

        for layer_idx, block in enumerate(self.h):
            x, new_layer_cache = block(x)
            if cache is not None:
                cache.write_prefix(layer_idx, *new_layer_cache)

        logits = self.ln_f(x)
        return logits

    def decode_one(
        self,
        indices: mx.array,
        layers: tuple[LayerCacheMLX, ...],
        position: mx.array,
    ) -> tuple[mx.array, tuple[LayerCacheMLX, ...]]:
        x = self.drop(self.wte(indices) + self.wpe(position.reshape((1,))))
        updated_layers = []
        for block, layer_cache in zip(self.h, layers, strict=True):
            x, updated_cache = block.decode_one(x, layer_cache, position)
            updated_layers.append(updated_cache)
        return self.ln_f(x), tuple(updated_layers)


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

    def __call__(self, x: mx.array) -> tuple[mx.array, LayerCacheMLX]:
        # Residual blocks: x + ...
        attention_output, new_cache = self.attn(self.ln_1(x))
        x = x + attention_output
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache

    def decode_one(
        self, x: mx.array, cache: LayerCacheMLX, position: mx.array
    ) -> tuple[mx.array, LayerCacheMLX]:
        attention_output, updated_cache = self.attn.decode_one(
            self.ln_1(x), cache, position
        )
        x = x + attention_output
        return x + self.mlp(self.ln_2(x)), updated_cache


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

    def __call__(self, x: mx.array) -> tuple[mx.array, LayerCacheMLX]:
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
        # To account for multihead attention, for the query, key and value matrices,
        # we break the embeddings in n_head parts, each one with head_dim components,
        # and transpose them so that each head is a 'batch' dimension and we operate
        # on all (B, n_head) matrices (T, head_dim) in parallel.
        # (B, T, C) -> (B, n_head, T, head_dim)
        query, key, value = self._project_qkv(x)

        # Each query scores every key, but can only see its own and earlier tokens.
        scores = (query @ key.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        query_positions = mx.arange(sequence_length)[:, None]
        key_positions = mx.arange(sequence_length)[None, :]
        visible = key_positions <= query_positions
        scores = mx.where(visible, scores, float("-inf"))
        weights = self.attn_dropout(mx.softmax(scores, axis=-1))
        output = weights @ value
        output = output.transpose(0, 2, 1, 3).reshape(
            batch_size,
            sequence_length,
            dim_embed,
        )
        output = self.c_proj(output)
        return self.resid_dropout(output), (key, value)

    def _project_qkv(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        batch_size, sequence_length, _ = x.shape
        query, key, value = mx.split(self.c_attn(x), 3, axis=-1)

        def split_heads(part: mx.array) -> mx.array:
            return part.reshape(
                batch_size, sequence_length, self.n_head, self.head_dim
            ).transpose(0, 2, 1, 3)

        return split_heads(query), split_heads(key), split_heads(value)

    def decode_one(
        self, x: mx.array, cache: LayerCacheMLX, position: mx.array
    ) -> tuple[mx.array, LayerCacheMLX]:
        query, key, value = self._project_qkv(x)
        key_cache = mx.slice_update(cache[0], key, position, (2,))
        value_cache = mx.slice_update(cache[1], value, position, (2,))

        # Only the new query is needed; it scores every cached key.
        scores = (query @ key_cache.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        visible = mx.arange(key_cache.shape[2]) <= position
        scores = mx.where(visible, scores, float("-inf"))
        weights = self.attn_dropout(mx.softmax(scores, axis=-1))
        output = weights @ value_cache
        output = output.transpose(0, 2, 1, 3).reshape(x.shape)
        return self.resid_dropout(self.c_proj(output)), (key_cache, value_cache)


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


def make_gpt2_train_step(
    model: GPT2MLXModel,
    optimizer: Optimizer,
    loss_fn: MLXLossFunction,
    *,
    compiled: bool = True,
    master_weights: bool = False,
) -> Callable[[mx.array, mx.array], mx.array]:
    def loss(model: nn.Module, inputs: mx.array, targets: mx.array) -> mx.array:
        logits = model(inputs).astype(mx.float32)
        return mx.mean(loss_fn(logits, targets))

    loss_and_grad = nn.value_and_grad(model, loss)
    master = None
    if master_weights:
        if model.lm_head.weight.dtype != mx.bfloat16:
            raise ValueError("FP32 master weights require a bfloat16 compute model")
        parameters = tree_map(lambda parameter: parameter, model.trainable_parameters())
        parameters["transformer"]["wte"].pop("weight")
        master = [tree_map(lambda parameter: parameter.astype(mx.float32), parameters)]

    def step(inputs: mx.array, targets: mx.array) -> mx.array:
        loss_value, grads = loss_and_grad(model, inputs, targets)
        embedding_grad = grads["transformer"]["wte"].pop("weight")
        grads["lm_head"]["weight"] += embedding_grad
        if master is None:
            optimizer.update(model, grads)
        else:
            gradients = tree_map(lambda gradient: gradient.astype(mx.float32), grads)
            master[0] = optimizer.apply_gradients(gradients, master[0])
            model.update(
                tree_map(lambda parameter: parameter.astype(mx.bfloat16), master[0])
            )
        model.transformer.wte.weight = model.lm_head.weight
        return loss_value

    if not compiled:
        return step

    state = [model.state, optimizer.state, mx.random.state]
    if master is not None:
        state.append(master)
    return partial(mx.compile, inputs=state, outputs=state)(step)


def iter_gpt2_train_steps(
    step: Callable[[mx.array, mx.array], mx.array],
    batches: Iterable[tuple[mx.array, mx.array]],
    state: list[Any],
    *,
    async_eval: bool = True,
) -> Iterator[tuple[mx.array, int]]:
    """Yield completed losses and token counts, with at most two steps in flight.

    In async mode, submit the next step before waiting for the previous one.
    Snapshot the state containers (without copying arrays) because each step
    replaces their contents. A yielded loss and its optimizer update are ready
    to read, so logging and progress updates do not introduce another GPU wait.
    """
    # ((loss, state_snapshot), token_count)
    pending: tuple[tuple[mx.array, list[Any]], int] | None = None
    try:
        for inputs, targets in batches:
            loss = step(inputs, targets)
            if not async_eval:
                mx.eval(loss, state)
                yield loss, targets.size
                continue

            outputs = tree_map(lambda value: value, (loss, state))
            mx.async_eval(outputs)
            if pending is not None:
                mx.eval(pending[0])
                yield pending[0][0], pending[1]
            pending = outputs, targets.size

        if pending is not None:
            outputs, tokens = pending
            mx.eval(outputs)
            yield outputs[0], tokens
    finally:
        # Drain updates on exhaustion, early close, or an exception.
        mx.eval(state)


def train_gpt2(
    model: GPT2MLXModel,
    dataloader: MLXDataLoader,
    optimizer: Optimizer,
    loss_fn: MLXLossFunction,
    epochs: int = 50,
    verbose: bool = True,
    compiled: bool = True,
    master_weights: bool = False,
    async_eval: bool = True,
    progress: bool = True,
) -> GPT2MLXModel:
    step = make_gpt2_train_step(
        model, optimizer, loss_fn, compiled=compiled, master_weights=master_weights
    )
    state = [model.state, optimizer.state, mx.random.state]
    total_batches = len(dataloader) if isinstance(dataloader, Sized) else None

    for e in range(epochs):
        dataloader.reset()
        model.train()
        epoch_start = perf_counter()
        epoch_loss = 0.0
        epoch_tokens = 0

        with tqdm(
            total=total_batches,
            desc=f"Epoch {e + 1}/{epochs}",
            unit="batch",
            mininterval=0.5,
            dynamic_ncols=True,
            disable=not progress,
            leave=False,
        ) as bar:
            for loss, tokens in iter_gpt2_train_steps(
                step, dataloader, state, async_eval=async_eval
            ):
                epoch_tokens += tokens
                if verbose:
                    epoch_loss += loss.item() * tokens
                bar.update()

        if epoch_tokens == 0:
            raise ValueError("cannot train on an empty dataloader")
        if verbose:
            elapsed = perf_counter() - epoch_start
            print(
                f"[Epoch {e + 1} / {epochs}]: "
                f"Train loss {epoch_loss / epoch_tokens:.4f}, "
                f"Elapsed time: {elapsed:.0f}, "
                f"{epoch_tokens / elapsed:.0f} tokens/sec"
                "]"
            )

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
        losses = loss_fn(logits.astype(mx.float32), targets)
        total_loss += mx.sum(losses)
        total_tokens += targets.size
        mx.eval(total_loss)

    dataloader.reset()

    if total_tokens == 0:
        raise ValueError("cannot evaluate an empty dataloader.")

    return total_loss / total_tokens
