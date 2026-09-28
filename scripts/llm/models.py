from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx

from neurnet.arch.llm.inference import InferenceRuntime
from neurnet.arch.llm.models.gpt2 import GPT2MLXModel, GPT2ModelType, GPT2Tokenizer
from neurnet.arch.llm.models.qwen import load_model_and_tokenizer
from neurnet.arch.llm.types import Tokenizer
from neurnet.device import Device

GPT2_MODEL_CHOICES = tuple(str(model_type) for model_type in GPT2ModelType)
QWEN_MODEL_CHOICES = ("qwen3",)
MODEL_CHOICES = (*QWEN_MODEL_CHOICES, *GPT2_MODEL_CHOICES)


def load_model_runtime(
    model_name: str,
    device: Device,
    *,
    compile: bool = False,
    reasoning: bool = False,
    models_dir: Path = Path("models"),
) -> InferenceRuntime:
    """Load a supported model behind the common inference runtime."""
    if model_name in QWEN_MODEL_CHOICES:
        model_type = "reasoning" if reasoning else "base"
        model, tokenizer = load_model_and_tokenizer(
            model_type,
            device.to_torch(),
            compile,
            models_dir / "qwen3",
        )
        return InferenceRuntime(
            model,
            tokenizer,
            device,
            eos_token_ids=_eos_token_ids(
                tokenizer,
                extra_tokens=("<|im_end|>", "<|endoftext|>"),
            ),
        )

    if model_name in GPT2_MODEL_CHOICES:
        if reasoning:
            raise ValueError("--reasoning is only supported by --model qwen3")

        mx.set_default_device(device.to_mlx())
        model_type = GPT2ModelType(model_name)
        model_dir = models_dir / "gpt2"
        model = GPT2MLXModel.from_pretrained(model_type, cache_dir=str(model_dir))
        if compile:
            model.compile()
        tokenizer = GPT2Tokenizer.from_pretrained(model_type, cache_dir=str(model_dir))
        return InferenceRuntime(
            model,
            tokenizer,
            device,
            eos_token_ids=_eos_token_ids(tokenizer),
        )

    raise ValueError(f"Unsupported model: {model_name}")


def _eos_token_ids(
    tokenizer: Tokenizer, *, extra_tokens: Sequence[str] = ()
) -> tuple[int, ...]:
    eos_token_ids: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        eos_token_ids.append(int(eos_token_id))

    for token in extra_tokens:
        token_ids = tokenizer.encode(token)
        if token_ids:
            eos_token_ids.append(int(token_ids[0]))

    return tuple(dict.fromkeys(eos_token_ids))
