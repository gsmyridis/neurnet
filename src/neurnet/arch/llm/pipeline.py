from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any, cast

import mlx.core as mx
import torch

from neurnet.arch.llm.kv_cache import KVCache
from neurnet.arch.llm.traits import (
    LanguageModel,
    LanguageModelMLX,
    LanguageModelTorch,
    TensorType,
    Tokenizer,
)

# ===-----------------------------------------------------------------------===
# Backend-agnostic inference pipeline for language model
# ===-----------------------------------------------------------------------===


def generate_token_stream(
    model: LanguageModel,
    token_ids: TensorType,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    *,
    use_kv_cache: bool = False,
) -> Iterator[int]:
    model.check_tensor_type(token_ids)

    if isinstance(model, LanguageModelMLX):
        assert isinstance(token_ids, mx.array)
        return generate_token_stream_mlx(
            model, token_ids, max_new_tokens, eos_token_id, use_kv_cache=use_kv_cache
        )

    elif isinstance(model, LanguageModelTorch):
        assert isinstance(token_ids, torch.Tensor)
        return generate_token_stream_torch(
            model, token_ids, max_new_tokens, eos_token_id, use_kv_cache=use_kv_cache
        )

    raise TypeError(f"Unsupported LanguageModel type: {type(model)}")


def generate_text_stream(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: Any,
    max_new_tokens: int,
    verbose: bool = False,
) -> str:
    encoded_prompt = tokenizer.encode(prompt)
    if isinstance(model, LanguageModelMLX):
        input_ids: TensorType = mx.array([encoded_prompt])
    elif isinstance(model, LanguageModelTorch):
        input_ids = torch.tensor(encoded_prompt, device=device).unsqueeze(0)
    else:
        raise TypeError(f"Unsupported LanguageModel type: {type(model)}")

    generated_ids: list[int] = []
    for token_id in generate_token_stream(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    ):
        generated_ids.append(token_id)

        if verbose:
            print(tokenizer.decode([token_id]), end="", flush=True)
    return tokenizer.decode(generated_ids)


# ===-----------------------------------------------------------------------===
# Inference pipeline for MLX language model
# ===-----------------------------------------------------------------------===


def _generate_token_stream_mlx_cached(
    model: LanguageModelMLX,
    token_ids: mx.array,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> Iterator[int]:
    model.eval()
    cache = KVCache[mx.array](n_layers=model.config().n_layers)
    model.reset_kv_cache()

    out = model(token_ids, cache=cache)[:, -1]
    for step in range(max_new_tokens):
        next_token = mx.argmax(out, axis=-1, keepdims=True)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        yield cast(int, next_token.item())

        if step + 1 < max_new_tokens:
            out = model(next_token, cache=cache)[:, -1]


def _generate_token_stream_mlx_not_cached(
    model: LanguageModelMLX,
    token_ids: mx.array,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> Iterator[int]:
    model.eval()

    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]
        next_token = mx.argmax(out, axis=-1, keepdims=True)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        yield cast(int, next_token.item())

        token_ids = mx.concat([token_ids, next_token], axis=1)


def generate_token_stream_mlx(
    model: LanguageModelMLX,
    token_ids: mx.array,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    *,
    use_kv_cache: bool = False,
) -> Iterator[int]:
    """
    Yields generated token IDs one-by-one in a stream, for an MLX model.
    """
    if use_kv_cache:
        return _generate_token_stream_mlx_cached(
            model,
            token_ids,
            max_new_tokens,
            eos_token_id,
        )

    return _generate_token_stream_mlx_not_cached(
        model,
        token_ids,
        max_new_tokens,
        eos_token_id,
    )


# ===-----------------------------------------------------------------------===
# Inference pipeline for PyTorch language model
# ===-----------------------------------------------------------------------===


@torch.inference_mode()
def _generate_token_stream_torch_cached(
    model: LanguageModelTorch,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> Iterator[int]:
    model.eval()
    cache = KVCache[torch.Tensor](n_layers=model.config().n_layers)
    model.reset_kv_cache()

    out = model(token_ids, cache=cache)[:, -1]
    for step in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

        yield cast(int, next_token.item())

        if step + 1 < max_new_tokens:
            out = model(next_token, cache=cache)[:, -1]


@torch.inference_mode()
def _generate_token_stream_torch_not_cached(
    model: LanguageModelTorch,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> Iterator[int]:
    model.eval()

    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

        yield cast(int, next_token.item())

        token_ids = torch.cat([token_ids, next_token], dim=1)


@torch.inference_mode()
def generate_token_stream_torch(
    model: LanguageModelTorch,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    *,
    use_kv_cache: bool = False,
) -> Iterator[int]:
    if use_kv_cache:
        return _generate_token_stream_torch_cached(
            model,
            token_ids,
            max_new_tokens,
            eos_token_id,
        )

    return _generate_token_stream_torch_not_cached(
        model,
        token_ids,
        max_new_tokens,
        eos_token_id,
    )


# ===-----------------------------------------------------------------------===
#
# ===-----------------------------------------------------------------------===


def _print_peak_memory_stats(device: str | torch.device) -> None:
    device = torch.device(device)
    backend = getattr(torch, device.type, None)
    if backend is None or not hasattr(backend, "is_available"):
        return
    if not backend.is_available():
        return

    sync_fn = getattr(backend, "synchronize", None)
    if callable(sync_fn):
        try:
            sync_fn(device=device)
        except TypeError:
            sync_fn()

    try:
        max_mem_bytes = backend.max_memory_allocated(device=device)
    except TypeError:
        max_mem_bytes = backend.max_memory_allocated()

    max_mem_gb = max_mem_bytes / (1024**3)
    print(f"Max {device.type.upper()} memory allocated: {max_mem_gb:.2f} GB")

    try:
        backend.reset_peak_memory_stats(device=device)
    except TypeError:
        backend.reset_peak_memory_stats()


def generate_stats(
    output_token_ids: torch.Tensor,
    tokenizer: object,
    start_time: float,
    end_time: float,
) -> None:
    # tokenizer is currently unused but retained for backward compatibility
    total_time = end_time - start_time
    print(f"\n\nTime: {total_time:.2f} sec")
    print(f"{int(output_token_ids.numel() / total_time)} tokens/sec")

    for name, backend in (
        ("CUDA", getattr(torch, "cuda", None)),
        ("XPU", getattr(torch, "xpu", None)),
    ):
        if backend is not None and backend.is_available():
            # Check whether we are actually using this backend
            device_type = output_token_ids.device.type
            if device_type != name.lower():
                warnings.warn(
                    f"{name} is available but tensors are on "
                    f"{device_type}. Memory stats may be 0."
                )

            # Synchronize if supported (important for async backends)
            if hasattr(backend, "synchronize"):
                backend.synchronize()

            max_mem_bytes = backend.max_memory_allocated()
            max_mem_gb = max_mem_bytes / (1024**3)
            print(f"Max {name} memory allocated: {max_mem_gb:.2f} GB")
            backend.reset_peak_memory_stats()
