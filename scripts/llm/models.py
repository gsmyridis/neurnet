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


def resolve_model_name(
    model_name: str | None,
    *,
    checkpoint_path: Path | None = None,
    hugging_face: str | None = None,
    reasoning: bool = False,
) -> str:
    """Resolve source defaults and reject incompatible model/source choices."""
    if checkpoint_path is not None and hugging_face is not None:
        raise ValueError("choose either --checkpoint-path or --hugging-face")
    if hugging_face is not None:
        if hugging_face not in GPT2_MODEL_CHOICES:
            raise ValueError("--hugging-face currently supports GPT-2 variants only")
        if model_name is not None and model_name != hugging_face:
            raise ValueError("--model must match the --hugging-face variant")
        model_name = hugging_face
    if model_name is None:
        model_name = "gpt2" if checkpoint_path is not None else "qwen3"
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"Unsupported model: {model_name}")
    if checkpoint_path is not None and model_name not in GPT2_MODEL_CHOICES:
        raise ValueError("--checkpoint-path currently supports GPT-2 MLX weights only")
    if reasoning and model_name not in QWEN_MODEL_CHOICES:
        raise ValueError("--reasoning is only supported by --model qwen3")
    return model_name


def load_model_runtime(
    model_name: str | None,
    device: Device,
    *,
    compile: bool = False,
    reasoning: bool = False,
    models_dir: Path = Path("models"),
    checkpoint_path: Path | None = None,
    hugging_face: str | None = None,
) -> InferenceRuntime:
    """Load a supported model behind the common inference runtime."""
    model_name = resolve_model_name(
        model_name,
        checkpoint_path=checkpoint_path,
        hugging_face=hugging_face,
        reasoning=reasoning,
    )
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
        mx.set_default_device(device.to_mlx())
        model_type = GPT2ModelType(model_name)
        model_dir = models_dir / "gpt2"
        if checkpoint_path is not None:
            model = GPT2MLXModel.from_checkpoint(
                checkpoint_path, model_type.get_config()
            )
        else:
            model = GPT2MLXModel.from_pretrained(model_type, cache_dir=str(model_dir))
        mx.eval(model.parameters())
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
