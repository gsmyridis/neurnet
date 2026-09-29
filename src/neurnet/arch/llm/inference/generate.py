from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
import torch

from neurnet.arch.llm.types import (
    LanguageModel,
    LanguageModelMLX,
    LanguageModelTorch,
    TensorType,
    Tokenizer,
)


@dataclass(frozen=True)
class GenerationPolicy:
    """Sampling and loop-prevention controls for autoregressive generation."""

    temperature: float = 0.7
    top_k: int | None = None
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    max_consecutive_repeats: int = 3
    repeat_ngram_size: int = 3

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be at least one when provided")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        if self.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be at least one")
        if self.max_consecutive_repeats < 0:
            raise ValueError("max_consecutive_repeats cannot be negative")
        if self.repeat_ngram_size < 0 or self.repeat_ngram_size == 1:
            raise ValueError("repeat_ngram_size must be zero or at least two")


# ===--------------------------------------------------------------------------===
# Backend-agnostic generation pipeline
# ===--------------------------------------------------------------------------===


def generate_token_stream(
    model: LanguageModel,
    token_ids: TensorType,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    *,
    use_kv_cache: bool = False,
    policy: GenerationPolicy | None = None,
) -> Iterator[int]:
    model.check_tensor_type(token_ids)

    if isinstance(model, LanguageModelMLX):
        assert isinstance(token_ids, mx.array)
        return generate_token_stream_mlx(
            model,
            token_ids,
            max_new_tokens,
            eos_token_id,
            use_kv_cache=use_kv_cache,
            policy=policy,
        )

    if isinstance(model, LanguageModelTorch):
        assert isinstance(token_ids, torch.Tensor)
        return generate_token_stream_torch(
            model,
            token_ids,
            max_new_tokens,
            eos_token_id,
            use_kv_cache=use_kv_cache,
            policy=policy,
        )

    raise TypeError(f"Unsupported LanguageModel type: {type(model)}")


def generate_text_stream(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: Any,
    max_new_tokens: int,
    verbose: bool = False,
    *,
    policy: GenerationPolicy | None = None,
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
        policy=policy,
    ):
        generated_ids.append(token_id)

        if verbose:
            print(tokenizer.decode([token_id]), end="", flush=True)
    return tokenizer.decode(generated_ids)


# ===--------------------------------------------------------------------------===
# Generation pipeline for MLX model
# ===--------------------------------------------------------------------------===


def _select_next_token_mlx(
    logits: mx.array,
    seen_token_ids: list[int],
    policy: GenerationPolicy | None,
) -> mx.array:
    if policy is None:
        return mx.argmax(logits, axis=-1, keepdims=True)

    logits = _apply_repetition_penalty_mlx(logits, seen_token_ids, policy)
    if policy.temperature == 0:
        return mx.argmax(logits, axis=-1, keepdims=True)

    logits = _top_k_filter_mlx(logits / policy.temperature, policy.top_k)
    logits = _top_p_filter_mlx(logits, policy.top_p)
    return mx.expand_dims(mx.random.categorical(logits, axis=-1), axis=-1)


def _apply_repetition_penalty_mlx(
    logits: mx.array,
    seen_token_ids: list[int],
    policy: GenerationPolicy,
) -> mx.array:
    if policy.repetition_penalty == 1 or not seen_token_ids:
        return logits

    token_ids = mx.array([sorted(set(seen_token_ids))])
    selected_logits = mx.take_along_axis(logits, token_ids, axis=-1)
    penalized_logits = mx.where(
        selected_logits < 0,
        selected_logits * policy.repetition_penalty,
        selected_logits / policy.repetition_penalty,
    )
    return mx.put_along_axis(logits, token_ids, penalized_logits, axis=-1)


def _top_p_filter_mlx(logits: mx.array, top_p: float) -> mx.array:
    if top_p == 1:
        return logits

    sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
    sorted_probabilities = mx.softmax(sorted_logits, axis=-1)
    cumulative_probabilities = mx.cumsum(sorted_probabilities, axis=-1)
    positions = mx.arange(logits.shape[-1])[None, :]
    keep = mx.logical_or(cumulative_probabilities <= top_p, positions == 0)
    filtered_sorted_logits = mx.where(keep, sorted_logits, float("-inf"))
    filtered_logits = mx.full(logits.shape, float("-inf"), dtype=logits.dtype)
    return mx.put_along_axis(
        filtered_logits,
        sorted_indices,
        filtered_sorted_logits,
        axis=-1,
    )


def _top_k_filter_mlx(logits: mx.array, top_k: int | None) -> mx.array:
    if top_k is None or top_k >= logits.shape[-1]:
        return logits

    sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
    top_indices = sorted_indices[:, :top_k]
    top_logits = mx.take_along_axis(logits, top_indices, axis=-1)
    filtered_logits = mx.full(logits.shape, float("-inf"), dtype=logits.dtype)
    return mx.put_along_axis(filtered_logits, top_indices, top_logits, axis=-1)


def _generate_token_stream_mlx_cached(
    model: LanguageModelMLX,
    token_ids: mx.array,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    *,
    policy: GenerationPolicy | None,
) -> Iterator[int]:
    model.eval()
    cache = model.create_kv_cache(
        batch_size=token_ids.shape[0],
        max_length=token_ids.shape[1] + max(0, max_new_tokens - 1),
    )
    seen_token_ids = [
        cast(int, token_ids[0, index].item()) for index in range(token_ids.shape[1])
    ]
    generated_ids: list[int] = []

    out = model(token_ids, cache=cache)[:, -1]
    for step in range(max_new_tokens):
        next_token = _select_next_token_mlx(out, seen_token_ids, policy)
        next_token_id = cast(int, next_token.item())

        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        if _should_stop_for_repetition(generated_ids, next_token_id, policy):
            break

        yield next_token_id
        generated_ids.append(next_token_id)
        seen_token_ids.append(next_token_id)

        if step + 1 < max_new_tokens:
            out = model(next_token, cache=cache)[:, -1]


def _generate_token_stream_mlx_not_cached(
    model: LanguageModelMLX,
    token_ids: mx.array,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    *,
    policy: GenerationPolicy | None,
) -> Iterator[int]:
    model.eval()
    seen_token_ids = [
        cast(int, token_ids[0, index].item()) for index in range(token_ids.shape[1])
    ]
    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]
        next_token = _select_next_token_mlx(out, seen_token_ids, policy)
        next_token_id = cast(int, next_token.item())

        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        if _should_stop_for_repetition(generated_ids, next_token_id, policy):
            break

        yield next_token_id
        generated_ids.append(next_token_id)
        seen_token_ids.append(next_token_id)
        token_ids = mx.concat([token_ids, next_token], axis=1)


def generate_token_stream_mlx(
    model: LanguageModelMLX,
    token_ids: mx.array,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    *,
    use_kv_cache: bool = False,
    policy: GenerationPolicy | None = None,
) -> Iterator[int]:
    """Yield generated token IDs one-by-one for an MLX model."""
    if use_kv_cache:
        return _generate_token_stream_mlx_cached(
            model,
            token_ids,
            max_new_tokens,
            eos_token_id,
            policy=policy,
        )

    return _generate_token_stream_mlx_not_cached(
        model,
        token_ids,
        max_new_tokens,
        eos_token_id,
        policy=policy,
    )


# ===--------------------------------------------------------------------------===
# Generation pipeline for Torch model
# ===--------------------------------------------------------------------------===


def _select_next_token_torch(
    logits: torch.Tensor,
    seen_token_ids: list[int],
    policy: GenerationPolicy | None,
) -> torch.Tensor:
    if policy is None:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = _apply_repetition_penalty_torch(logits, seen_token_ids, policy)
    if policy.temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = _top_k_filter_torch(logits / policy.temperature, policy.top_k)
    logits = _top_p_filter_torch(logits, policy.top_p)
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


def _apply_repetition_penalty_torch(
    logits: torch.Tensor,
    seen_token_ids: list[int],
    policy: GenerationPolicy,
) -> torch.Tensor:
    if policy.repetition_penalty == 1 or not seen_token_ids:
        return logits

    token_ids = torch.tensor(
        sorted(set(seen_token_ids)),
        device=logits.device,
        dtype=torch.long,
    )
    selected_logits = logits.index_select(-1, token_ids)
    penalized_logits = torch.where(
        selected_logits < 0,
        selected_logits * policy.repetition_penalty,
        selected_logits / policy.repetition_penalty,
    )
    return logits.scatter(
        -1,
        token_ids.expand(logits.shape[0], -1),
        penalized_logits,
    )


def _top_p_filter_torch(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p == 1:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    keep = cumulative_probabilities <= top_p
    keep[..., 0] = True
    filtered_sorted_logits = torch.where(
        keep,
        sorted_logits,
        torch.full_like(sorted_logits, float("-inf")),
    )
    return torch.full_like(logits, float("-inf")).scatter(
        -1,
        sorted_indices,
        filtered_sorted_logits,
    )


def _top_k_filter_torch(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k >= logits.shape[-1]:
        return logits

    top_logits, top_indices = torch.topk(logits, k=top_k, dim=-1)
    return torch.full_like(logits, float("-inf")).scatter(
        -1,
        top_indices,
        top_logits,
    )


@torch.inference_mode()
def _generate_token_stream_torch_cached(
    model: LanguageModelTorch,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    *,
    policy: GenerationPolicy | None,
) -> Iterator[int]:
    model.eval()
    cache = model.create_kv_cache(
        batch_size=token_ids.shape[0],
        max_length=token_ids.shape[1] + max(0, max_new_tokens - 1),
        device=token_ids.device,
    )
    seen_token_ids = [int(token_id) for token_id in token_ids.flatten().tolist()]
    generated_ids: list[int] = []

    out = model(token_ids, cache=cache)[:, -1]
    for step in range(max_new_tokens):
        next_token = _select_next_token_torch(out, seen_token_ids, policy)
        next_token_id = cast(int, next_token.item())

        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        if _should_stop_for_repetition(generated_ids, next_token_id, policy):
            break

        yield next_token_id
        generated_ids.append(next_token_id)
        seen_token_ids.append(next_token_id)

        if step + 1 < max_new_tokens:
            out = model(next_token, cache=cache)[:, -1]


@torch.inference_mode()
def _generate_token_stream_torch_not_cached(
    model: LanguageModelTorch,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    *,
    policy: GenerationPolicy | None,
) -> Iterator[int]:
    model.eval()
    seen_token_ids = [int(token_id) for token_id in token_ids.flatten().tolist()]
    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]
        next_token = _select_next_token_torch(out, seen_token_ids, policy)
        next_token_id = cast(int, next_token.item())

        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        if _should_stop_for_repetition(generated_ids, next_token_id, policy):
            break

        yield next_token_id
        generated_ids.append(next_token_id)
        seen_token_ids.append(next_token_id)
        token_ids = torch.cat([token_ids, next_token], dim=1)


@torch.inference_mode()
def generate_token_stream_torch(
    model: LanguageModelTorch,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    *,
    use_kv_cache: bool = False,
    policy: GenerationPolicy | None = None,
) -> Iterator[int]:
    if use_kv_cache:
        return _generate_token_stream_torch_cached(
            model,
            token_ids,
            max_new_tokens,
            eos_token_id,
            policy=policy,
        )

    return _generate_token_stream_torch_not_cached(
        model,
        token_ids,
        max_new_tokens,
        eos_token_id,
        policy=policy,
    )


def _should_stop_for_repetition(
    generated_ids: list[int],
    next_token_id: int,
    policy: GenerationPolicy | None,
) -> bool:
    if policy is None:
        return False

    if policy.max_consecutive_repeats:
        repeat_count = policy.max_consecutive_repeats
        if len(generated_ids) >= repeat_count and all(
            token_id == next_token_id for token_id in generated_ids[-repeat_count:]
        ):
            return True

    ngram_size = policy.repeat_ngram_size
    if ngram_size == 0 or len(generated_ids) + 1 < 2 * ngram_size:
        return False

    candidate_ngram = tuple((generated_ids + [next_token_id])[-ngram_size:])
    return any(
        tuple(generated_ids[start : start + ngram_size]) == candidate_ngram
        for start in range(len(generated_ids) - ngram_size + 1)
    )


# ===--------------------------------------------------------------------------===
# Generation stats for Torch model
# ===--------------------------------------------------------------------------===


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
            device_type = output_token_ids.device.type
            if device_type != name.lower():
                warnings.warn(
                    f"{name} is available but tensors are on "
                    f"{device_type}. Memory stats may be 0."
                )

            if hasattr(backend, "synchronize"):
                backend.synchronize()

            max_mem_bytes = backend.max_memory_allocated()
            max_mem_gb = max_mem_bytes / (1024**3)
            print(f"Max {name} memory allocated: {max_mem_gb:.2f} GB")
            backend.reset_peak_memory_stats()
