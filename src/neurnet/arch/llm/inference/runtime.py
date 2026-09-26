from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import mlx.core as mx
import torch

from neurnet.arch.llm.types import (
    LanguageModel,
    LanguageModelMLX,
    LanguageModelTorch,
    TensorType,
    Tokenizer,
)
from neurnet.device import Device

from .generate import GenerationPolicy, generate_token_stream


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 256
    use_kv_cache: bool = True
    policy: GenerationPolicy = field(default_factory=GenerationPolicy)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    elapsed_seconds: float


class InferenceRuntime:
    """Backend-independent text generation for one loaded language model."""

    def __init__(
        self,
        model: LanguageModel,
        tokenizer: Tokenizer,
        device: Device,
        *,
        eos_token_ids: Sequence[int] = (),
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._eos_token_ids = frozenset(eos_token_ids)
        self._context_length = model.config().context_length
        model.eval()

    @property
    def context_length(self) -> int:
        return self._context_length

    def set_seed(self, seed: int) -> None:
        """Seed the active backend's generator for reproducible generation."""
        if seed < 0:
            raise ValueError("seed cannot be negative")

        if isinstance(self._model, LanguageModelMLX):
            mx.random.seed(seed)
            return

        if isinstance(self._model, LanguageModelTorch):
            torch.manual_seed(seed)
            return

        raise TypeError(f"Unsupported LanguageModel type: {type(self._model)}")

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        """Generate one completion and optionally stream safe decoded text chunks."""
        input_ids = self._make_input_ids(self._tokenizer.encode(prompt))
        input_ids = _trim_input_tensor(
            input_ids,
            context_length=self._context_length,
            max_new_tokens=config.max_new_tokens,
        )

        start_time = time.perf_counter()
        token_ids: list[int] = []
        text_streamer = _DecodedTextStreamer(self._tokenizer, on_text)

        for token_id in generate_token_stream(
            model=self._model,
            token_ids=input_ids,
            max_new_tokens=config.max_new_tokens,
            use_kv_cache=config.use_kv_cache,
            policy=config.policy,
        ):
            if token_id in self._eos_token_ids:
                break
            token_ids.append(token_id)
            text_streamer.push(token_id)

        text = text_streamer.finish()
        return GenerationResult(
            text=text,
            token_ids=tuple(token_ids),
            elapsed_seconds=time.perf_counter() - start_time,
        )

    def _make_input_ids(self, token_ids: Sequence[int]) -> TensorType:
        if isinstance(self._model, LanguageModelMLX):
            mx.set_default_device(self._device.to_mlx())
            return mx.array([token_ids])

        if isinstance(self._model, LanguageModelTorch):
            return torch.tensor(token_ids, device=self._device.to_torch()).unsqueeze(0)

        raise TypeError(f"Unsupported LanguageModel type: {type(self._model)}")


class _DecodedTextStreamer:
    """Avoid emitting replacement characters from incomplete byte-level tokens."""

    def __init__(
        self, tokenizer: Tokenizer, on_text: Callable[[str], None] | None
    ) -> None:
        self._tokenizer = tokenizer
        self._on_text = on_text
        self._token_ids: list[int] = []
        self._emitted_text = ""

    def push(self, token_id: int) -> None:
        self._token_ids.append(token_id)
        decoded = self._tokenizer.decode(self._token_ids)
        safe_text = decoded.split("\ufffd", maxsplit=1)[0]
        self._emit_suffix(safe_text)

    def finish(self) -> str:
        decoded = self._tokenizer.decode(self._token_ids)
        self._emit_suffix(decoded)
        return decoded

    def _emit_suffix(self, text: str) -> None:
        if not text.startswith(self._emitted_text):
            return

        suffix = text.removeprefix(self._emitted_text)
        if suffix and self._on_text is not None:
            self._on_text(suffix)
        self._emitted_text = text


def _trim_input_tensor(
    input_ids: TensorType,
    *,
    context_length: int,
    max_new_tokens: int,
) -> TensorType:
    if not 0 < max_new_tokens < context_length:
        raise ValueError(
            "max_new_tokens must be greater than zero and smaller than the model "
            "context length"
        )

    keep_length = max(1, context_length - max_new_tokens)
    if input_ids.shape[1] > keep_length:
        return input_ids[:, -keep_length:]
    return input_ids
